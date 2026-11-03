import os
import sys
import argparse
import torch
import shutil
import datetime
import tempfile
import inspect
from omegaconf import OmegaConf

# 1. Environment variables to prevent common Windows multi-threading & Lhotse loader bugs
os.environ["LHOTSE_DISABLE"] = "1"
os.environ["NEMO_USE_LEGACY_DATALOADER"] = "1"

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    print("[ERROR] NVIDIA NeMo is not installed in the current Python environment.")
    print("Please install it using: pip install nemo_toolkit[asr]")
    sys.exit(1)

def force_windows_safe_dataloader_cfg(cfg):
    """
    Recursively patches the model configuration to disable multi-processing
    dataloaders and Lhotse, preventing crashes on Windows.
    """
    if not OmegaConf.is_config(cfg):
        return

    targets = {
        "use_lhotse": False,
        "pretokenize": False,
        "num_workers": 0,
        "pin_memory": False
    }
    
    OmegaConf.set_struct(cfg, False)

    for key, target_val in targets.items():
        try:
            if not OmegaConf.is_missing(cfg, key) and OmegaConf.select(cfg, key) is not None:
                OmegaConf.update(cfg, key, target_val, merge=False)
        except Exception:
            pass

    if hasattr(cfg, "items"):
        try:
            for k, child in cfg.items():
                if not OmegaConf.is_missing(cfg, k):
                    force_windows_safe_dataloader_cfg(child)
        except Exception:
            pass

def get_audio_duration(file_path):
    """Retrieves audio duration using soundfile or ffprobe."""
    # Try soundfile
    try:
        import soundfile as sf
        return sf.info(file_path).duration
    except Exception:
        pass
    
    # Try ffprobe
    try:
        import subprocess
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)]
        return float(subprocess.check_output(probe_cmd).decode().strip())
    except Exception:
        pass
    
    return 0.0

def split_audio_ffmpeg(file_path, temp_dir, chunk_length_s=300, overlap_s=10):
    """
    Splits the audio file into chunks of chunk_length_s with overlap_s.
    Converts audio to 16kHz, mono, WAV format (pcm_s16le) required by NeMo.
    """
    import subprocess
    
    duration = get_audio_duration(file_path)
    if duration == 0:
        print("[WARNING] Could not determine audio duration. Processing full file in one chunk.")
        # Fallback: normalize the whole file
        output_path = os.path.join(temp_dir, "normalized_full.wav")
        cmd = [
            'ffmpeg', '-y',
            '-i', str(file_path),
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return [output_path]
        except Exception as e:
            print(f"[ERROR] Normalization failed: {e}. Proceeding with raw file.")
            return [str(file_path)]

    print(f"Total audio duration: {duration:.2f} seconds")
    
    # Step 1: Normalize entire file to 16kHz mono WAV first to ensure sample-accurate seeking
    normalized_full = os.path.join(temp_dir, "normalized_full.wav")
    print("Normalizing full audio to 16kHz mono PCM WAV...")
    cmd_norm = [
        'ffmpeg', '-y',
        '-i', str(file_path),
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        normalized_full
    ]
    try:
        subprocess.run(cmd_norm, check=True, capture_output=True)
    except Exception as e:
        print(f"[ERROR] Full audio normalization failed: {e}. Falling back to direct input.")
        normalized_full = str(file_path)

    # If the file is shorter than a chunk, just return normalized_full
    if duration <= chunk_length_s:
        return [normalized_full]

    # Chunking logic from normalized WAV
    chunks = []
    start = 0
    idx = 0
    while start < duration:
        output_path = os.path.join(temp_dir, f"chunk_{idx:03d}.wav")
        actual_length = chunk_length_s + overlap_s
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-t', str(actual_length),
            '-i', normalized_full,
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            output_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                chunks.append(output_path)
        except Exception as e:
            print(f"[WARNING] Failed to extract chunk {idx}: {e}")
        start += chunk_length_s
        idx += 1
        
    return chunks

def extract_word_timestamps(hypothesis, chunk_offset, tokenizer, frame_duration=0.08):
    """Extracts word-level timestamps from NeMo transcription hypothesis."""
    # Check modern NeMo 2.0+ timestamp dict first
    if hasattr(hypothesis, 'timestamp') and isinstance(hypothesis.timestamp, dict):
        if 'word' in hypothesis.timestamp and hypothesis.timestamp['word']:
            words = []
            for w in hypothesis.timestamp['word']:
                word_text = w.get('word', '').strip()
                if not word_text:
                    continue
                s = chunk_offset + int(w['start_offset']) * frame_duration
                e = chunk_offset + int(w['end_offset']) * frame_duration
                if e <= s:
                    e = s + frame_duration
                words.append({
                    'word': word_text,
                    'start': round(s, 3),
                    'end': round(e, 3)
                })
            if words:
                return words

    # Legacy NeMo timestep attribute
    timesteps = getattr(hypothesis, 'timestep', None)
    if not timesteps:
        return None
        
    try:
        y_seq = hypothesis.y_sequence if hasattr(hypothesis, 'y_sequence') else None
        if y_seq is None: 
            return None
        
        tokens = [tokenizer.ids_to_tokens([int(tid)])[0] for tid in y_seq]
        timesteps = hypothesis.timestep
        
        if len(tokens) > len(timesteps):
            tokens = tokens[len(tokens)-len(timesteps):]
        
        words = []
        current_word = ""
        current_start = None
        
        # Unicode space character for SentencePiece is usually \u2581 or regular space
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
        print(f"[WARNING] Native timestamp extraction failed: {e}")
        return None

def interpolate_word_timestamps(text, start_time, end_time):
    """Interpolates word timestamps linearly across a segment if native timestamps are missing."""
    words = text.split()
    if not words: 
        return []
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

def group_words_into_segments(words_with_times, max_words=8, max_dur=4.5, pause_thresh=0.55):
    """Groups words into natural subtitle segments based on pauses and max words/duration."""
    if not words_with_times:
        return []

    segments = []
    current_chunk = []
    
    for w in words_with_times:
        if not current_chunk:
            current_chunk.append(w)
            continue
            
        prev_w = current_chunk[-1]
        pause = w['start'] - prev_w['end']
        cur_dur = w['end'] - current_chunk[0]['start']
        cur_count = len(current_chunk)
        
        # Split condition: significant pause (> 0.55s), or reached max words or max duration
        if pause >= pause_thresh or cur_count >= max_words or cur_dur >= max_dur:
            start_t = current_chunk[0]['start']
            end_t = current_chunk[-1]['end']
            
            # Add small comfort padding if pause allows
            if pause > 0.15:
                end_t = min(end_t + 0.12, w['start'] - 0.05)
            if end_t - start_t < 1.0 and (w['start'] - start_t >= 1.0):
                end_t = min(start_t + 1.0, w['start'] - 0.05)
                
            text = " ".join([cw['word'] for cw in current_chunk]).strip()
            if text:
                segments.append({"start": round(start_t, 3), "end": round(end_t, 3), "text": text})
            current_chunk = [w]
        else:
            current_chunk.append(w)
            
    if current_chunk:
        start_t = current_chunk[0]['start']
        end_t = current_chunk[-1]['end']
        if end_t - start_t < 1.0:
            end_t = start_t + 1.0
        text = " ".join([cw['word'] for cw in current_chunk]).strip()
        if text:
            segments.append({"start": round(start_t, 3), "end": round(end_t, 3), "text": text})
            
    return segments

def format_timestamp(seconds):
    """Formats float seconds into SRT timestamp format (HH:MM:SS,mmm)."""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def format_srt(segments):
    """Converts a list of segment dicts to a formatted SRT string."""
    srt_lines = []
    for i, segment in enumerate(segments, 1):
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        srt_lines.append(f"{i}")
        srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        srt_lines.append(f"{text.strip()}\n")
    return "\n".join(srt_lines)

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to SRT using NeMo FastConformer.")
    parser.add_argument(
        "--audio", 
        type=str, 
        required=True, 
        help="Path to the input audio file (.wav, .mp3, .m4a, etc.)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Path to output .srt file (defaults to audio file name + .srt)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo", 
        help="Path to local .nemo model or HuggingFace identifier"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--chunk-len", 
        type=int, 
        default=300, 
        help="Chunk length in seconds for long audio splitting"
    )
    parser.add_argument(
        "--overlap", 
        type=int, 
        default=10, 
        help="Overlap in seconds between chunks"
    )
    parser.add_argument(
        "--max-words", 
        type=int, 
        default=9, 
        help="Maximum words per SRT line"
    )
    
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"[ERROR] Audio file not found: {args.audio}")
        sys.exit(1)

    # Determine default output file path
    output_srt = args.output
    if not output_srt:
        base, _ = os.path.splitext(args.audio)
        output_srt = f"{base}.srt"

    print(f"Device: {args.device}")
    device = torch.device(args.device)

    # Load Model
    print(f"Loading NeMo model: {args.model} ...")
    try:
        if os.path.exists(args.model):
            try:
                model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
                    restore_path=args.model,
                    map_location=device
                )
            except Exception:
                model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=args.model,
                    map_location=device
                )
        else:
            try:
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=args.model,
                    map_location=device
                )
            except Exception:
                model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)
                if hasattr(model, "to"):
                    model.to(device)
    except Exception as e:
        print(f"[ERROR] Failed loading model: {e}")
        sys.exit(1)

    print(f"Loaded model: {model.__class__.__name__}")

    # Patch model configuration for Windows safety
    if os.name == "nt" and hasattr(model, "cfg"):
        print("Applying Windows compatibility patches to config...")
        force_windows_safe_dataloader_cfg(model.cfg)

    # Calculate frame duration dynamically
    stride = getattr(getattr(model.cfg, 'preprocessor', None), 'window_stride', 0.01) if hasattr(model, 'cfg') else 0.01
    subsampling = getattr(getattr(model.cfg, 'encoder', None), 'subsampling_factor', 8) if hasattr(model, 'cfg') else 8
    frame_dur = stride * subsampling
    print(f"Computed frame duration: {frame_dur:.4f}s (stride={stride}, subsampling={subsampling})")

    # Change to RNNT Greedy Decoding strategy if it's a hybrid model
    if hasattr(model, "change_decoding_strategy"):
        print("Setting decoding strategy to rnnt_greedy with preserve_alignments=True, compute_timestamps=True...")
        model.change_decoding_strategy(
            decoder_type="rnnt", 
            decoding_cfg={
                "strategy": "greedy",
                "preserve_alignments": True,
                "compute_timestamps": True
            }
        )

    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    # Create temporary directory for audio chunking
    temp_dir = tempfile.mkdtemp(prefix="nemo_srt_")
    print(f"Temporary workspace: {temp_dir}")
    
    try:
        # Split audio into normalized 16kHz mono chunks
        print("Preprocessing audio file...")
        chunks = split_audio_ffmpeg(
            args.audio, 
            temp_dir, 
            chunk_length_s=args.chunk_len, 
            overlap_s=args.overlap
        )
        print(f"Created {len(chunks)} chunks for processing.")

        timestamped_segments = []
        all_filtered_words = []
        sig = inspect.signature(model.transcribe)
        arg_name = "paths2audio_files" if "paths2audio_files" in sig.parameters else "audio"

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Reset PyTorch CUDA Cache to prevent memory leakage
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
            transcribe_kwargs = {
                arg_name: [chunk],
                "batch_size": 1,
                "num_workers": 0
            }
            if "return_hypotheses" in sig.parameters:
                transcribe_kwargs["return_hypotheses"] = True

            with torch.inference_mode():
                res = model.transcribe(**transcribe_kwargs)
                
            if res and isinstance(res, list) and len(res) > 0:
                hypothesis = res[0]
                text = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)
                
                # Calculate chunk time boundaries
                chunk_start_shifted = i * args.chunk_len
                chunk_end_shifted = chunk_start_shifted + args.chunk_len + args.overlap
                
                # Extract word level timestamps
                words = extract_word_timestamps(hypothesis, chunk_start_shifted, model.tokenizer, frame_duration=frame_dur)
                
                # Fallback to linear interpolation if timestamps are missing
                if not words or len(words) == 0:
                    words = interpolate_word_timestamps(text, chunk_start_shifted, chunk_end_shifted)
                
                # Filter out overlap between adjacent chunks
                official_start = i * args.chunk_len
                official_end = (i + 1) * args.chunk_len
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

                # Collect words from chunk
                if filtered_words:
                    all_filtered_words.extend(filtered_words)
            else:
                print(f"[WARNING] Chunk {i+1} returned no transcript.")

        # Group all accumulated words across chunks into natural subtitle segments
        if all_filtered_words:
            all_filtered_words.sort(key=lambda x: x['start'])
            timestamped_segments = group_words_into_segments(
                all_filtered_words, 
                max_words=args.max_words,
                max_dur=4.5,
                pause_thresh=0.55
            )

        # Save to SRT file
        if timestamped_segments:
            srt_content = format_srt(timestamped_segments)
            with open(output_srt, "w", encoding="utf-8") as f:
                f.write(srt_content)
            print(f"\n[SUCCESS] SRT file written successfully to: {output_srt}")
        else:
            print("[ERROR] No transcription segments found. SRT file was not generated.")

    finally:
        # Clean up temporary directory and files
        print("Cleaning up temporary audio chunks...")
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
