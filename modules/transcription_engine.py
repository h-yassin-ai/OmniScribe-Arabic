import torch
import gc
import os
import time
import shutil
import subprocess
import glob
import tempfile
import datetime
import traceback
from pydub import AudioSegment
from .utils import logger, DOWNLOADS_DIR, split_audio_robust

# Bypass SSL verification for Hugging Face downloads (corporate/Windows certificates fix)
try:
    import requests
    from huggingface_hub import configure_http_backend
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.verify = False
        return session
    configure_http_backend(backend_factory=backend_factory)
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logger.info("SSL verification bypassed for Hugging Face.")
except Exception as e:
    pass

try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logger.error("NeMo Toolkit not found.")

import inspect
from omegaconf import OmegaConf
import numpy as np

# =====================================================================
# === HF Transformers ASR Pipeline Batch Squeeze Monkeypatch ===
# =====================================================================
try:
    from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
    
    _orig_postprocess = AutomaticSpeechRecognitionPipeline.postprocess

    def patched_postprocess(self, model_outputs, **kwargs):
        try:
            return _orig_postprocess(self, model_outputs, **kwargs)
        except ValueError as e:
            if "cannot select an axis to squeeze out which has size not equal to one" in str(e):
                logger.info("Interception: Applying robust batch squeeze fix to AutomaticSpeechRecognitionPipeline.")
                
                final_items = []
                key = "logits" if self.type == "ctc_with_lm" else "tokens"
                stride = None
                for outputs in model_outputs:
                    if outputs[key].dtype in (torch.bfloat16, torch.float16):
                        items = outputs[key].to(torch.float32).numpy()
                    else:
                        items = outputs[key].numpy()
                    stride = outputs.get("stride", None)
                    if stride is not None and self.type in {"ctc", "ctc_with_lm"}:
                        total_n, left, right = stride
                        right_n = total_n - right
                        items = items[:, left:right_n]
                    final_items.append(items)

                items = np.concatenate(final_items, axis=1)
                
                # Safe squeeze only if first dim is 1
                if len(items.shape) > 0 and items.shape[0] == 1:
                    items = items.squeeze(0)
                
                text = ""
                if self.type == "ctc_with_lm":
                    decoder_kwargs = kwargs.get("decoder_kwargs", {}) or {}
                    if len(items.shape) == 2 and items.shape[0] > 1:
                        beams_list = [self.decoder.decode_beams(seq, **decoder_kwargs) for seq in items]
                        text = " ".join([b[0][0] for b in beams_list if b])
                    else:
                        beams = self.decoder.decode_beams(items, **decoder_kwargs)
                        text = beams[0][0]
                else:
                    skip_special_tokens = self.type != "ctc"
                    decode_kwargs = {"group_tokens": False} if self.type == "tdt" else {}
                    
                    if len(items.shape) == 2 and items.shape[0] > 1:
                        decoded_texts = []
                        for seq in items:
                            decoded_texts.append(self.tokenizer.decode(seq, skip_special_tokens=skip_special_tokens, **decode_kwargs))
                        text = " ".join([t.strip() for t in decoded_texts if t.strip()])
                    else:
                        if len(items.shape) == 2:
                            items = items.squeeze(0)
                        text = self.tokenizer.decode(items, skip_special_tokens=skip_special_tokens, **decode_kwargs)

                return {"text": text, "chunks": []}
            else:
                raise e

    AutomaticSpeechRecognitionPipeline.postprocess = patched_postprocess
    logger.info("AutomaticSpeechRecognitionPipeline.postprocess successfully patched for robust batch decoding.")
except Exception as patch_err:
    logger.warning(f"ASR Pipeline postprocess patch failed to register: {patch_err}")

def merge_tokens_to_words(tokens):
    """
    Groups individual SentencePiece tokens back into words based on prefix space characters.
    """
    words = []
    current_word = ""
    word_start = None
    word_end = None
    
    for t in tokens:
        token_text = t.get('token', '')
        start = t.get('start', 0.0)
        end = t.get('end', 0.0)
        
        # Space indicator check (SentencePiece U+2581 or typical spaces)
        starts_new = token_text.startswith(" ") or token_text.startswith(" ") or token_text.startswith("_") or (words and token_text.startswith(" "))
        
        cleaned_token = token_text.replace(" ", "").replace(" ", "").replace("_", "")
        
        if starts_new:
            if current_word.strip():
                words.append({
                    'word': current_word.strip(),
                    'start': word_start,
                    'end': word_end
                })
            current_word = cleaned_token
            word_start = start
            word_end = end
        else:
            if not current_word:
                word_start = start
            current_word += cleaned_token
            word_end = end
            
    if current_word.strip():
        words.append({
            'word': current_word.strip(),
            'start': word_start,
            'end': word_end
        })
        
    return words

def normalize_arabic(text):
    # Remove diacritics
    import re
    diacritics = re.compile(r'[\u064B-\u0652]')
    text = diacritics.sub('', text)
    # Normalize hamza and common variations
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    return text.strip()

def align_words_dp(cohere_words, nemotron_words_with_times, chunk_start=0.0, chunk_end=30.0):
    n = len(cohere_words)
    m = len(nemotron_words_with_times)
    
    if n == 0:
        return []
    if m == 0:
        # Interpolate everything over the chunk
        d = (chunk_end - chunk_start) / n
        return [{'word': w, 'start': chunk_start + i*d, 'end': chunk_start + (i+1)*d} for i, w in enumerate(cohere_words)]
        
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    parent = [[None] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        dp[i][0] = i * 1.5
        parent[i][0] = (i - 1, 0, 'delete')
    for j in range(1, m + 1):
        dp[0][j] = j * 1.5
        parent[0][j] = (0, j - 1, 'insert')
        
    for i in range(1, n + 1):
        cw = cohere_words[i - 1]
        for j in range(1, m + 1):
            nw = nemotron_words_with_times[j - 1]['word']
            
            norm_cw = normalize_arabic(cw)
            norm_nw = normalize_arabic(nw)
            
            if norm_cw == norm_nw:
                match_cost = 0.0
            else:
                match_cost = 1.0 # Substitution penalty
                
            cost_match = dp[i-1][j-1] + match_cost
            cost_delete = dp[i-1][j] + 1.5
            cost_insert = dp[i][j-1] + 1.5
            
            min_cost = min(cost_match, cost_delete, cost_insert)
            dp[i][j] = min_cost
            
            if min_cost == cost_match:
                parent[i][j] = (i - 1, j - 1, 'match')
            elif min_cost == cost_delete:
                parent[i][j] = (i - 1, j, 'delete')
            else:
                parent[i][j] = (i, j - 1, 'insert')
                
    # Backtrack
    curr_i, curr_j = n, m
    matches = [] # list of (cohere_idx, nemotron_idx)
    while curr_i > 0 or curr_j > 0:
        p = parent[curr_i][curr_j]
        if p is None:
            break
        prev_i, prev_j, action = p
        if action == 'match':
            matches.append((curr_i - 1, curr_j - 1))
        curr_i, curr_j = prev_i, prev_j
        
    matches.reverse()
    
    # Build aligned words list
    aligned_words = [{'word': cw, 'start': None, 'end': None} for cw in cohere_words]
    for ci, ni in matches:
        aligned_words[ci]['start'] = nemotron_words_with_times[ni]['start']
        aligned_words[ci]['end'] = nemotron_words_with_times[ni]['end']
        
    # Interpolate unmatched blocks
    unmatched_indices = []
    for i in range(n):
        if aligned_words[i]['start'] is None:
            unmatched_indices.append(i)
        else:
            if unmatched_indices:
                interpolate_block(aligned_words, unmatched_indices, chunk_start, chunk_end)
                unmatched_indices = []
                
    if unmatched_indices:
        interpolate_block(aligned_words, unmatched_indices, chunk_start, chunk_end)
        
    return aligned_words

def interpolate_block(aligned_words, indices, chunk_start, chunk_end):
    k = len(indices)
    first_idx = indices[0]
    last_idx = indices[-1]
    
    t_L = chunk_start
    if first_idx > 0:
        for idx in range(first_idx - 1, -1, -1):
            if aligned_words[idx]['end'] is not None:
                t_L = aligned_words[idx]['end']
                break
                
    t_R = chunk_end
    if last_idx < len(aligned_words) - 1:
        for idx in range(last_idx + 1, len(aligned_words)):
            if aligned_words[idx]['start'] is not None:
                t_R = aligned_words[idx]['start']
                break
                
    total_dur = t_R - t_L
    if total_dur < 0.01:
        total_dur = 0.01
    segment_dur = total_dur / (k + 1)
    
    for step, idx in enumerate(indices):
        aligned_words[idx]['start'] = t_L + (step + 0.5) * segment_dur
        aligned_words[idx]['end'] = aligned_words[idx]['start'] + segment_dur


class TranscriptionEngine:
    def __init__(self, model_path, device='cuda'):
        if not NEMO_AVAILABLE:
            raise ImportError("NeMo Toolkit is not installed.")
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.processor = None
        
        self.alignment_model = None
        self.alignment_processor = None
        self.alignment_pipe = None
        
        # Load Model
        self.load_model()
        
    def get_alignment_model(self):
        if self.alignment_model is None:
            logger.info("Loading Nemotron alignment model for Cohere timestamps...")
            from transformers import pipeline, AutoProcessor
            model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
            device_idx = -1
            if self.device.type == "cuda":
                if ":" in str(self.device):
                    device_idx = int(str(self.device).split(":")[-1])
                else:
                    device_idx = 0
            
            # Load alignment pipeline
            self.alignment_pipe = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=device_idx,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            self.alignment_model = self.alignment_pipe.model
            self.alignment_processor = AutoProcessor.from_pretrained(model_id)
            logger.info("Nemotron alignment model loaded successfully.")
        return self.alignment_model, self.alignment_processor
        
    def load_model(self):
        self.is_transformers = False
        self.is_hybrid = False
        self.processor = None
        try:
            # Check if it is a Hugging Face hub repository path
            if "/" in self.model_path and not os.path.exists(self.model_path):
                if "cohere" in self.model_path.lower():
                    self.is_transformers = True
                    logger.info(f"Loading HF transformers model: {self.model_path}...")
                    from transformers import pipeline, AutoProcessor
                    device_idx = -1
                    if self.device.type == "cuda":
                        if ":" in str(self.device):
                            device_idx = int(str(self.device).split(":")[-1])
                        else:
                            device_idx = 0
                    self.model = pipeline(
                        "automatic-speech-recognition",
                        model=self.model_path,
                        device=device_idx,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                    )
                    try:
                        self.processor = AutoProcessor.from_pretrained(self.model_path)
                    except Exception as pe:
                        logger.warning(f"Could not load AutoProcessor for {self.model_path}: {pe}")
                    logger.info("Transformers pipeline loaded successfully.")
                    return
                else:
                    logger.info(f"Loading NeMo model from HuggingFace: {self.model_path}...")
                    try:
                        try:
                            self.model = nemo_asr.models.ASRModel.from_pretrained(
                                model_name=self.model_path,
                                map_location=self.device
                            )
                        except TypeError:
                            self.model = nemo_asr.models.ASRModel.from_pretrained(
                                model_name=self.model_path
                            )
                            if hasattr(self.model, "to"):
                                self.model.to(self.device)
                    except Exception as nemo_e:
                        logger.warning(f"NeMo load failed: {nemo_e}. Falling back to HF transformers pipeline...")
                        self.is_transformers = True
                        from transformers import pipeline, AutoProcessor
                        device_idx = -1
                        if self.device.type == "cuda":
                            if ":" in str(self.device):
                                device_idx = int(str(self.device).split(":")[-1])
                            else:
                                device_idx = 0
                        self.model = pipeline(
                            "automatic-speech-recognition",
                            model=self.model_path,
                            device=device_idx,
                            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
                        )
                        try:
                            self.processor = AutoProcessor.from_pretrained(self.model_path)
                        except Exception as pe:
                            logger.warning(f"Could not load AutoProcessor for {self.model_path}: {pe}")
                        logger.info("Loaded NeMo model using HF transformers pipeline fallback.")
                        return
            else:
                # Local .nemo model file
                if self.device.type == "cuda":
                    logger.info(f"Loading local model from {os.path.basename(self.model_path)} on {self.device}...")
                    try:
                        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
                            restore_path=self.model_path, 
                            map_location=self.device
                        )
                    except Exception as e:
                        logger.warning(f"Failed loading as EncDecHybridRNNTCTCBPEModel: {e}. Trying general ASRModel.")
                        self.model = nemo_asr.models.ASRModel.restore_from(
                            restore_path=self.model_path,
                            map_location=self.device
                        )
                else:
                    logger.info(f"Loading local model from {os.path.basename(self.model_path)} on CPU...")
                    try:
                        self.model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
                            restore_path=self.model_path, 
                            map_location="cpu"
                        )
                    except Exception as e:
                        logger.warning(f"Failed loading as EncDecHybridRNNTCTCBPEModel on CPU: {e}. Trying general ASRModel.")
                        self.model = nemo_asr.models.ASRModel.restore_from(
                            restore_path=self.model_path,
                            map_location="cpu"
                        )
            
            # NeMo post-load configuration
            logger.info(f"Model loaded: {self.model.__class__.__name__}")
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            
            # Verify Device
            try:
                 param_device = next(self.model.parameters()).device
                 logger.info(f"VERIFIED: Model parameters are on {param_device}")
            except Exception as ve:
                 logger.warning(f"Could not verify model device explicitly: {ve}")

            self.is_hybrid = isinstance(self.model, nemo_asr.models.EncDecHybridRNNTCTCBPEModel)

            # CRITICAL WINDOWS FIX: Valid configuration manipulation via OmegaConf
            if os.name == "nt" and hasattr(self.model, 'cfg'):
                self._force_windows_safe_dataloader_cfg(self.model.cfg)
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def _force_windows_safe_dataloader_cfg(self, cfg):
        """
        NUCLEAR: Forces num_workers=0 and AGGRESSIVELY disables Lhotse/Pretokenization
        by recursively searching the entire config tree.
        """
        logger.info("NUCLEAR: Patching model config recursively: num_workers=0, Lhotse OFF, pretokenize OFF...")
        
        def recursive_patch(node):
            if not OmegaConf.is_config(node):
                return
            
            # Target keys
            targets = {
                "use_lhotse": False,
                "pretokenize": False,
                "num_workers": 0,
                "pin_memory": False
            }
            
            # Use structured=False to allow modifications to potentially "missing" or structured configs
            OmegaConf.set_struct(node, False)

            for key, target_val in targets.items():
                try:
                    # Only update if the key exists AND isn't missing
                    # select() returns None if key doesn't exist, but we must check if it's "missing"
                    if not OmegaConf.is_missing(node, key) and OmegaConf.select(node, key) is not None:
                        OmegaConf.update(node, key, target_val, merge=False)
                except Exception:
                    pass
            
            # Recurse into children safely
            if hasattr(node, "items"):
                try:
                    for k, child in node.items():
                        # Skip processing keys that are flagged as missing to avoid MissingMandatoryValue
                        if not OmegaConf.is_missing(node, k):
                            recursive_patch(child)
                except Exception:
                    pass

        # 1. Recursive config patch
        recursive_patch(cfg)

        # 2. Global environment variables
        os.environ["LHOTSE_DISABLE"] = "1"
        os.environ["NEMO_USE_LEGACY_DATALOADER"] = "1"
        
        # 3. Monkeypatching NeMo's dataloader factory (The Final Frontier)
        try:
            from nemo.collections.asr.data import audio_to_text
            if hasattr(audio_to_text, 'get_lhotse_dataloader'):
                logger.debug("Monkeypatching nemo.collections.asr.data.audio_to_text.get_lhotse_dataloader to return None")
                audio_to_text.get_lhotse_dataloader = lambda *args, **kwargs: None
        except Exception as e:
            logger.warning(f"Monkeypatch failed (expected if NeMo version differs): {e}")

    def format_srt(self, segments):
        """Converts list of (start, end, text) to SRT format."""
        srt_lines = []
        for i, segment in enumerate(segments, 1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            srt_lines.append(f"{i}")
            srt_lines.append(f"{self._format_timestamp(start)} --> {self._format_timestamp(end)}")
            srt_lines.append(f"{text.strip()}\n")
        return "\n".join(srt_lines)

    def _format_timestamp(self, seconds):
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int(td.microseconds / 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def _extract_word_timestamps(self, hypothesis, chunk_offset, frame_duration=0.04):
        """
        Attempts to extract word-level timestamps from a NeMo hypothesis.
        """
        if not hasattr(hypothesis, 'timestep') or not hypothesis.timestep:
            return None
            
        try:
            tokenizer = self.model.tokenizer
            # Some models have y_sequence, others just have a list of tokens/ids
            y_seq = hypothesis.y_sequence if hasattr(hypothesis, 'y_sequence') else None
            if y_seq is None: return None
            
            # Convert IDs to tokens
            tokens = [tokenizer.ids_to_tokens([int(tid)])[0] for tid in y_seq]
            # Strip BOS/EOS markers if present (usually at start/end)
            timesteps = hypothesis.timestep
            
            # Align if lengths differ (BOS often doesn't have a timestep)
            if len(tokens) > len(timesteps):
                tokens = tokens[len(tokens)-len(timesteps):]
            
            words = []
            current_word = ""
            current_start = None
            
            # Space marker for SentencePiece is '\u2581' ( )
            for token, step in zip(tokens, timesteps):
                is_new_word = token.startswith('\u2581') or token.startswith(' ')
                
                if is_new_word and current_word:
                    words.append({
                        'word': current_word.strip(),
                        'start': chunk_offset + current_start * frame_duration,
                        'end': chunk_offset + step * frame_duration
                    })
                    current_word = token.replace('\u2581', '').replace(' ', '')
                    current_start = step
                else:
                    if current_start is None:
                        current_start = step
                    current_word += token.replace('\u2581', '').replace(' ', '')
            
            if current_word:
                words.append({
                    'word': current_word.strip(),
                    'start': chunk_offset + current_start * frame_duration,
                    'end': chunk_offset + timesteps[-1] * frame_duration
                })
            return words
        except Exception as e:
            logger.warning(f"Word timestamp extraction failed: {e}")
            return None

    def _interpolate_word_timestamps(self, text, start_time, end_time):
        """Fallback: linear interpolation of words over the chunk duration."""
        words = text.split()
        if not words: return []
        duration = end_time - start_time
        avg_word_dur = duration / len(words)
        
        result = []
        for i, word in enumerate(words):
            result.append({
                'word': word,
                'start': start_time + i * avg_word_dur,
                'end': start_time + (i + 1) * avg_word_dur
            })
        return result

    def _group_words_into_segments(self, words_with_times, max_words=8):
        """Groups individual words into clusters of max_words for SRT entries."""
        segments = []
        for i in range(0, len(words_with_times), max_words):
            chunk = words_with_times[i:i + max_words]
            if not chunk: continue
            text = " ".join([w['word'] for w in chunk])
            start = chunk[0]['start']
            end = chunk[-1]['end']
            segments.append({"start": start, "end": end, "text": text})
        return segments

    def transcribe(self, audio_path=None, strategy="rnnt_greedy", pre_split_chunks=None, chunk_length_s=300, overlap_s=10):
        if audio_path is None and pre_split_chunks is None:
            return ["Error: No audio source provided", ""]

        # 1. Prepare Chunks
        temp_dir = None
        chunks = []
        if pre_split_chunks:
            chunks = pre_split_chunks
        elif audio_path:
            # Use centralized robust splitter with overlap
            chunks, temp_dir = split_audio_robust(audio_path, chunk_length_s=chunk_length_s, overlap_s=overlap_s)
        
        full_transcript = []
        timestamped_segments = []
        
        try:
            if getattr(self, "is_transformers", False):
                # 2a. Process Chunks via Transformers pipeline
                for i, chunk in enumerate(chunks):
                    try:
                        file_path = chunk.file_path if hasattr(chunk, 'file_path') else chunk
                        
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        logger.debug(f"Processing Transformers Chunk {i+1}/{len(chunks)}: {file_path}")
                        
                        is_nemotron = "nemotron" in str(self.model_path).lower()
                        
                        if hasattr(chunk, 'start_s') and hasattr(chunk, 'end_s'):
                            chunk_start_shifted = chunk.start_s
                            chunk_end_shifted = chunk.end_s
                        else:
                            chunk_start_shifted = i * chunk_length_s
                            chunk_end_shifted = chunk_start_shifted + chunk_length_s + overlap_s

                        words = []
                        text = ""

                        # If Nemotron model and processor is loaded, try native durations decoding
                        if is_nemotron and self.processor is not None and hasattr(self.model, "model"):
                            try:
                                from transformers.audio_utils import load_audio
                                model_device = self.model.model.device
                                model_dtype = self.model.model.dtype
                                
                                audio_data = load_audio(file_path, sampling_rate=self.processor.feature_extractor.sampling_rate)
                                inputs = self.processor(
                                    audio_data,
                                    sampling_rate=self.processor.feature_extractor.sampling_rate,
                                    language="ar-AR",
                                    return_tensors="pt"
                                ).to(model_device, dtype=model_dtype)
                                
                                with torch.inference_mode():
                                    output = self.model.model.generate(**inputs, return_dict_in_generate=True)
                                    
                                text_decoded, timestamps = self.processor.decode(
                                    output.sequences,
                                    durations=output.durations,
                                    skip_special_tokens=True
                                )
                                
                                text = text_decoded[0] if isinstance(text_decoded, list) else text_decoded
                                tokens = timestamps[0] if (timestamps and len(timestamps) > 0) else []
                                raw_words = merge_tokens_to_words(tokens)
                                
                                for rw in raw_words:
                                    words.append({
                                        'word': rw['word'],
                                        'start': chunk_start_shifted + rw['start'],
                                        'end': chunk_start_shifted + rw['end']
                                    })
                            except Exception as native_ts_err:
                                logger.warning(f"Native Nemotron timestamp generation failed: {native_ts_err}. Falling back to pipeline...")
                                words = []
                                text = ""

                        # Fallback to pipeline + linear interpolation
                        if not words:
                            generate_kwargs = {}
                            if is_nemotron:
                                generate_kwargs["language"] = "ar-AR"
                                
                            res = self.model(file_path, generate_kwargs=generate_kwargs)
                            if isinstance(res, dict):
                                text = res.get("text", "")
                            elif isinstance(res, list) and len(res) > 0:
                                text = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
                            else:
                                text = str(res)
                            
                            if "cohere" in str(self.model_path).lower():
                                try:
                                    align_model, align_processor = self.get_alignment_model()
                                    
                                    from transformers.audio_utils import load_audio
                                    audio_data = load_audio(file_path, sampling_rate=align_processor.feature_extractor.sampling_rate)
                                    
                                    model_device = align_model.device
                                    model_dtype = align_model.dtype
                                    
                                    inputs = align_processor(
                                        audio_data,
                                        sampling_rate=align_processor.feature_extractor.sampling_rate,
                                        language="ar-AR",
                                        return_tensors="pt"
                                    ).to(model_device, dtype=model_dtype)
                                    
                                    with torch.inference_mode():
                                        output = align_model.generate(**inputs, return_dict_in_generate=True)
                                        
                                    text_nemotron_decoded, timestamps_nemotron = align_processor.decode(
                                        output.sequences,
                                        durations=output.durations,
                                        skip_special_tokens=True
                                    )
                                    
                                    tokens_nemotron = timestamps_nemotron[0] if (timestamps_nemotron and len(timestamps_nemotron) > 0) else []
                                    raw_words_nemotron = merge_tokens_to_words(tokens_nemotron)
                                    
                                    cohere_words = text.strip().split()
                                    
                                    aligned_words = align_words_dp(
                                        cohere_words, 
                                        raw_words_nemotron, 
                                        chunk_start=0.0, 
                                        chunk_end=chunk_end_shifted - chunk_start_shifted
                                    )
                                    
                                    for aw in aligned_words:
                                        aw['start'] += chunk_start_shifted
                                        aw['end'] += chunk_start_shifted
                                        
                                    words = aligned_words
                                except Exception as align_err:
                                    logger.warning(f"Failed to align Cohere text with Nemotron timestamps: {align_err}. Falling back to linear interpolation...")
                                    words = self._interpolate_word_timestamps(text, chunk_start_shifted, chunk_end_shifted)
                            else:
                                words = self._interpolate_word_timestamps(text, chunk_start_shifted, chunk_end_shifted)
                        
                        if hasattr(chunk, 'start_s'):
                            # VAD chunk: keep all words
                            filtered_words = words
                        else:
                            # Fixed interval chunk: apply boundary filtering
                            official_start = i * chunk_length_s
                            official_end = (i + 1) * chunk_length_s
                            is_last_chunk = (i == len(chunks) - 1)
                            
                            filtered_words = []
                            for w in words:
                                if w['start'] < 0: w['start'] = 0.0
                                if w['end'] < 0: w['end'] = 0.01

                                if i == 0:
                                    if w['start'] < official_end:
                                        filtered_words.append(w)
                                else:
                                    if w['start'] >= official_start:
                                        if is_last_chunk or w['start'] < official_end:
                                            filtered_words.append(w)
                        
                        if filtered_words:
                            segments = self._group_words_into_segments(filtered_words, max_words=9)
                            timestamped_segments.extend(segments)
                            
                            chunk_text = " ".join([w['word'] for w in filtered_words])
                            full_transcript.append(chunk_text)
                            
                        logger.debug(f"Transformers Chunk {i+1} success. Kept {len(filtered_words)} words.")
                    except Exception as chunk_e:
                        logger.error(f"Error in Transformers Chunk {i+1}: {chunk_e}")
                        full_transcript.append(f"[Chunk {i+1} Error]")
                
                final_text = " ".join(full_transcript)
                final_srt = self.format_srt(timestamped_segments)
                logger.info("Transformers transcription merge complete.")
                return [final_text, final_srt, timestamped_segments]

            # 2b. Process Chunks (Universal NeMo Call)
            sig = inspect.signature(self.model.transcribe)
            arg_name = "paths2audio_files" if "paths2audio_files" in sig.parameters else "audio"

            for i, chunk in enumerate(chunks):
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    logger.debug(f"Processing Chunk {i+1}/{len(chunks)}: {chunk}")
                    
                    with torch.inference_mode():
                        file_path = chunk.file_path if hasattr(chunk, 'file_path') else chunk
                        kwargs = {arg_name: [file_path]}
                        if "return_hypotheses" in sig.parameters:
                            kwargs["return_hypotheses"] = True
                        if "batch_size" in sig.parameters:
                            kwargs["batch_size"] = 1
                        if "num_workers" in sig.parameters:
                            kwargs["num_workers"] = 0 
                        
                        # Configure Strategy (Hybrid Model Syntax)
                        if strategy == "rnnt_beam":
                            if hasattr(self.model, "change_decoding_strategy"):
                                # High Precision Beam Search
                                self.model.change_decoding_strategy(
                                    decoder_type="rnnt", 
                                    decoding_cfg={
                                        "strategy": "beam",
                                        "beam": {
                                            "beam_size": 4,
                                            "return_best_hypothesis": True,
                                            "score_norm": True
                                        }
                                    }
                                )
                        elif strategy == "rnnt_greedy":
                            if hasattr(self.model, "change_decoding_strategy"):
                                self.model.change_decoding_strategy(
                                    decoder_type="rnnt", 
                                    decoding_cfg={"strategy": "greedy"}
                                )
                        elif strategy == "ctc_greedy":
                            if hasattr(self.model, "change_decoding_strategy"):
                                self.model.change_decoding_strategy(
                                    decoder_type="ctc", 
                                    decoding_cfg={"strategy": "greedy"}
                                )
                        # Legacy/Fallback mappings
                        elif strategy == "rnnt":
                            if hasattr(self.model, "change_decoding_strategy"):
                                self.model.change_decoding_strategy(decoder_type="rnnt")
                        elif strategy == "greedy":
                            if hasattr(self.model, "change_decoding_strategy"):
                                self.model.change_decoding_strategy(decoder_type="ctc")
                        
                        if "use_lhotse" in sig.parameters: kwargs["use_lhotse"] = False
                        if "pretokenize" in sig.parameters: kwargs["pretokenize"] = False
                        if "pin_memory" in sig.parameters: kwargs["pin_memory"] = False

                        res = self.model.transcribe(**kwargs)
                    
                    if res and isinstance(res, list) and len(res) > 0:
                        hypothesis = res[0]
                        text = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)
                        
                        if not text or text.strip() == "":
                             logger.warning(f"Chunk {i+1} returned an empty transcription.")
                             # We still continue to handle potentially empty chunks gracefully
                        
                        # Timestamp Processing
                        if hasattr(chunk, 'start_s') and hasattr(chunk, 'end_s'):
                            chunk_start_shifted = chunk.start_s
                            chunk_end_shifted = chunk.end_s
                        else:
                            lead_in_s = 0.0
                            chunk_start_shifted = i * chunk_length_s
                            chunk_end_shifted = chunk_start_shifted + chunk_length_s + overlap_s
                        
                        # 1. Try real timestamps
                        words = self._extract_word_timestamps(hypothesis, chunk_start_shifted)
                        
                        # 2. Fallback to interpolation
                        if not words or len(words) == 0:
                            words = self._interpolate_word_timestamps(text, chunk_start_shifted, chunk_end_shifted)
                        
                        # 3. Deduplication & Bounds Clipping
                        if hasattr(chunk, 'start_s'):
                            # VAD chunk: keep all words
                            filtered_words = words
                        else:
                            # Fixed interval chunk: boundary filtering
                            official_start = i * chunk_length_s
                            official_end = (i + 1) * chunk_length_s
                            is_last_chunk = (i == len(chunks) - 1)
                            
                            filtered_words = []
                            for w in words:
                                if w['start'] < 0: w['start'] = 0.0
                                if w['end'] < 0: w['end'] = 0.01

                                if i == 0:
                                    if w['start'] < official_end:
                                        filtered_words.append(w)
                                else:
                                    if w['start'] >= official_start:
                                        if is_last_chunk or w['start'] < official_end:
                                            filtered_words.append(w)
                        
                        if filtered_words:
                            # 4. Group into 8-10 word segments
                            segments = self._group_words_into_segments(filtered_words, max_words=9)
                            timestamped_segments.extend(segments)
                            
                            chunk_text = " ".join([w['word'] for w in filtered_words])
                            full_transcript.append(chunk_text)
                            
                        logger.debug(f"Chunk {i+1} success. Adjusted for 1s lead-in. Kept {len(filtered_words)} words.")
                    else:
                        logger.warning(f"Chunk {i+1} returned NO hypothesis object.")
                        full_transcript.append(f"[Chunk {i+1} Missing Result]")
                    
                    # Cleanup VRAM immediately
                    del res
                except Exception as chunk_e:
                    logger.error(f"Error in Chunk {i+1} of {len(chunks)}: {traceback.format_exc()}")
                    full_transcript.append(f"[Chunk {i+1} Error]")
                
            # 3. Merge Results
            final_text = " ".join(full_transcript)
            final_srt = self.format_srt(timestamped_segments)
            logger.info("Transcription merge complete.")
            return [final_text, final_srt, timestamped_segments]
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return [f"Error: {str(e)}"]
        finally:
            # Safe cleanup: Ignore errors on Windows if file is held
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            torch.cuda.empty_cache()

    def unload(self):
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
