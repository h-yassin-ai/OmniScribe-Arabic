import os
import sys
import tempfile
import logging
import traceback
import ssl

# =====================================================================
# === 1. CRITICAL: Process-Wide SSL Verification Bypass (MUST BE FIRST) ===
# =====================================================================
# Disable SSL verification globally at the lowest Python API level
original_create_default_context = ssl.create_default_context
def new_create_default_context(*args, **kwargs):
    context = original_create_default_context(*args, **kwargs)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context
ssl.create_default_context = new_create_default_context
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

try:
    import requests
    original_send = requests.Session.send
    def new_send(self, request, **kwargs):
        kwargs['verify'] = False
        return original_send(self, request, **kwargs)
    requests.Session.send = new_send
except Exception:
    pass

try:
    import httpx
    original_client_init = httpx.Client.__init__
    def new_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        original_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = new_client_init

    original_async_client_init = httpx.AsyncClient.__init__
    def new_async_client_init(self, *args, **kwargs):
        kwargs['verify'] = False
        original_async_client_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = new_async_client_init
except Exception:
    pass

# Safe console output for non-ASCII/Arabic characters on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# =====================================================================
# === 2. CRITICAL: Hardened Environment Setup ===
# =====================================================================
os.environ['LHOTSE_DISABLE'] = '1'
os.environ['NEMO_USE_LEGACY_DATALOADER'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['HF_HOME'] = r'J:\UnivHm_NextJS\Al-Kutub-Automator\.cache\huggingface'

# =====================================================================
# === 3. CRITICAL: Windows File Lock Fixes ===
# =====================================================================
if sys.platform == 'win32':
    import shutil
    import datetime
    
    # Patch TemporaryDirectory._cleanup to prevent WinError 32 crashes on exit
    def _robust_rmtree(cls, name, ignore_errors=False):
        try:
            shutil.rmtree(name, ignore_errors=True)
        except Exception:
            pass
    tempfile.TemporaryDirectory._rmtree = classmethod(_robust_rmtree)
    
    # Make rmtree globally tolerant
    _orig_rmtree = shutil.rmtree
    def rmtree_no_fail(path, *args, **kwargs):
        kwargs["ignore_errors"] = True
        try:
            return _orig_rmtree(path, *args, **kwargs)
        except Exception:
            pass
    shutil.rmtree = rmtree_no_fail

# =====================================================================
# === 4. Precision Logging Setup ===
# =====================================================================
from modules.utils import logger, MODELS_DIR, OUTPUTS_DIR, get_output_path, split_audio_robust

# Force precision logging
precision_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
for handler in logger.handlers:
    handler.setFormatter(precision_formatter)
logger.setLevel(logging.DEBUG)

# Try to find Node.js for yt-dlp
try:
    node_path = shutil.which("node")
    if node_path:
        os.environ['YDL_JS_RUNTIME'] = node_path
        logger.info(f"YDL_JS_RUNTIME set to: {node_path}")
except Exception:
    pass

def global_exception_handler(exctype, value, tb):
    """Log any uncaught exception before exiting."""
    error_msg = "".join(traceback.format_exception(exctype, value, tb))
    
    # Custom 403 hint
    if "403" in error_msg:
        logger.critical("DETECTED 403 FORBIDDEN. Hint: Proof-of-Origin (PO Token) or fresh cookies required.")
        print("\n⚠️  ERREUR 403 DÉTECTÉE : YouTube bloque la connexion. Assurez-vous d'avoir mis à jour 'po_token.txt' ou 'cookies.txt'.", file=sys.stderr)
        
    logger.critical(f"FATAL UNCAUGHT EXCEPTION:\n{error_msg}")
    print(f"\n❌ FATAL ERROR:\n{error_msg}", file=sys.stderr)

sys.excepthook = global_exception_handler

# =====================================================================
# === 5. Load Configurations from .env on Startup ===
# =====================================================================
env_token = ""
env_api_id = ""
env_api_hash = ""
env_phone = ""

env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if key == "HF_TOKEN":
                        env_token = val
                        os.environ["HF_TOKEN"] = val
                    elif key == "TG_API_ID":
                        env_api_id = val
                    elif key == "TG_API_HASH":
                        env_api_hash = val
                    elif key == "TG_PHONE":
                        env_phone = val
    except Exception:
        pass

def save_env_values(hf_token, tg_api_id, tg_api_hash, tg_phone):
    existing_keys = {}
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.split("=", 1)
                        existing_keys[k.strip()] = v.strip()
        except Exception:
            pass
            
    # Update keys
    if hf_token is not None: existing_keys["HF_TOKEN"] = hf_token.strip()
    if tg_api_id is not None: existing_keys["TG_API_ID"] = tg_api_id.strip()
    if tg_api_hash is not None: existing_keys["TG_API_HASH"] = tg_api_hash.strip()
    if tg_phone is not None: existing_keys["TG_PHONE"] = tg_phone.strip()
    
    try:
        with open(env_path, "w", encoding="utf-8") as f:
            for k, v in existing_keys.items():
                f.write(f"{k}={v}\n")
        return "✅ Configuration enregistrée avec succès dans le .env !"
    except Exception as e:
        return f"❌ Erreur d'enregistrement : {e}"

# =====================================================================
# === 6. Remaining Imports ===
# =====================================================================
from modules.ocr_common import configure_transformers_compat
configure_transformers_compat()
from modules.qari_ocr import initialize_unsloth_runtime
initialize_unsloth_runtime()

import gradio as gr

import glob
import gc
import torch
import subprocess
import time
import datetime
import threading
import queue
import concurrent.futures
from modules.media_handler import get_video_info, download_audio, scan_local_folder
from modules.transcription_engine import TranscriptionEngine

# --- Helper to scan for models ---
def list_available_models():
    """Finds .nemo files and adds HF presets."""
    presets = [
        "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "CohereLabs/cohere-transcribe-arabic-07-2026"
    ]
    root_models = glob.glob("*.nemo")
    sub_models = glob.glob(os.path.join(MODELS_DIR, "*.nemo"))
    local_models = root_models + sub_models
    return presets + local_models

def get_available_gpus():
    """Returns list of available CUDA devices."""
    import torch
    if not torch.cuda.is_available():
        return []
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def get_device_choices():
    choices = [(f"GPU {index} — {torch.cuda.get_device_name(index)}", f"GPU {index}")
               for index in range(torch.cuda.device_count())]
    if len(choices) > 1:
        choices.append(("Tous les GPU disponibles", "All Available GPUs"))
    choices.append(("CPU", "CPU"))
    return choices

# =====================================================================
# === 7. Pipeline Logic ===
# =====================================================================

class VideoItem:
    def __init__(self, video_data):
        self.data = video_data
        self.wav_path = None
        self.chunks_dir = None
        self.chunks = []
        self.transcription = ""
        self.status = "Pending"
        self.error = None
        self.model_results = {}

def batch_download_phase(videos, proxy, progress_callback):
    """Phase 1: Download all videos in parallel"""
    results = []
    
    def _download_task(video):
        try:
            path = download_audio(video['url'], video_id=video['id'], proxy=proxy)
            return (video, path, None)
        except Exception as e:
            return (video, None, str(e))

    max_workers = min(4, max(1, len(videos)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_task, v): v for v in videos}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            video, path, err = future.result()
            progress_callback(i + 1, len(videos), f"Downloading: {video['title']}")
            
            item = VideoItem(video)
            if path:
                item.wav_path = path
                item.status = "Downloaded"
            else:
                item.status = "Failed Download"
                item.error = err
            results.append(item)
            
    return results

def batch_split_phase(items, chunk_length_s, progress_callback, overlap_s=10, split_method="VAD (silence intelligent)"):
    """Phase 2: Split audio using VAD or fixed interval splitting"""
    total = len([x for x in items if x.wav_path])
    current = 0
    
    for item in items:
        if not item.wav_path: continue
        
        try:
            if split_method == "VAD (silence intelligent)":
                from modules.utils import split_audio_vad
                chunks, temp_dir = split_audio_vad(item.wav_path)
            else:
                chunks, temp_dir = split_audio_robust(item.wav_path, chunk_length_s=chunk_length_s, overlap_s=overlap_s)
                
            item.chunks = chunks
            item.chunks_dir = temp_dir
            item.status = "Split Ready"
        except Exception as e:
            logger.error(f"Split failed for {item.data['title']}: {e}")
            item.status = "Failed Split"
            item.error = str(e)
            
        current += 1
        progress_callback(current, total, f"Splitting: {item.data['title']}")
            
    return items

class GPUTranscriberThread(threading.Thread):
    def __init__(self, device_id, model_path, queue, results_list, progress_callback, chunk_length_s):
        super().__init__()
        self.device_id = device_id
        self.model_path = model_path
        self.queue = queue
        self.results_list = results_list
        self.progress_call = progress_callback
        self.chunk_length_s = chunk_length_s
        self.overlap_s = 15
        self.engine = None
        self.error = None
        self.total_chunks_processed = 0
        
    def run(self):
        try:
            if "cuda" in str(self.device_id) and torch.cuda.is_available():
                device_idx = int(str(self.device_id).split(":")[-1])
                torch.cuda.set_device(device_idx)
                logger.info(f"[{self.device_id}] Explicitly set CUDA device {device_idx} for worker thread.")
            elif "cuda" in str(self.device_id):
                logger.warning(f"[{self.device_id}] CUDA requested but not available. Falling back to CPU behavior.")

            logger.info(f"[{self.device_id}] Starting model load from: {self.model_path}")
            if self.progress_call: self.progress_call(f"[{self.device_id}] Loading Model...")
            
            self.engine = TranscriptionEngine(self.model_path, device=self.device_id)
            logger.info(f"[{self.device_id}] TranscriptionEngine initialized successfully.")
        except Exception as e:
            err_msg = f"[{self.device_id}] Model Load Failed: {e}\n{traceback.format_exc()}"
            self.error = err_msg
            logger.error(err_msg)
            if self.progress_call: self.progress_call(f"[{self.device_id}] Model Load Failed")
            return

        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            
            try:
                if self.progress_call: self.progress_call(f"[{self.device_id}] Transcribing: {item.data['title']} ({len(item.chunks)} chunks)")
                
                strategy = getattr(self, "strategy", "greedy")
                res = self.engine.transcribe(
                    pre_split_chunks=item.chunks, 
                    strategy=strategy, 
                    chunk_length_s=self.chunk_length_s,
                    overlap_s=getattr(self, "overlap_s", 15)
                )
                
                transcription_text = ""
                transcription_srt = ""
                timestamped_segments = []
                if isinstance(res, list) and len(res) >= 3:
                    transcription_text, transcription_srt, timestamped_segments = res[0], res[1], res[2]
                elif isinstance(res, list) and len(res) >= 2:
                    transcription_text, transcription_srt = res[0], res[1]
                elif isinstance(res, list) and len(res) == 1:
                    transcription_text = res[0]
                
                item.model_results[self.model_path] = {
                    "text": transcription_text,
                    "srt": transcription_srt,
                    "segments": timestamped_segments
                }
                item.transcription = transcription_text
                item.status = "Completed"
                    
                model_clean = self.model_path.split("/")[-1].replace(".nemo", "")
                base_name = f"{item.data['title']} [{item.data['id']}]_{model_clean}"
                subfolder = getattr(self, "subfolder", None)
                
                # Save TXT
                txt_path = get_output_path(base_name, "txt", subfolder=subfolder)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(transcription_text)
                self.results_list.append(str(txt_path))
                
                # Save SRT
                if transcription_srt:
                    srt_path = get_output_path(base_name, "srt", subfolder=subfolder)
                    with open(srt_path, "w", encoding="utf-8") as f:
                        f.write(transcription_srt)
                    self.results_list.append(str(srt_path))
                
            except Exception as e:
                item.status = "Failed Transcription"
                item.error = f"{e}\n{traceback.format_exc()}" if 'traceback' in globals() else str(e)
                logger.error(f"[{self.device_id}] Error transcribing {item.data['title']}: {item.error}")
            finally:
                torch.cuda.empty_cache()
                self.queue.task_done()
        
        if self.engine:
            self.engine.unload()

def process_content_batch(source_type, source_input, model_paths, strategy, device_id, chunk_length_s, overlap_s, proxy_url, hf_token_input, split_method, progress=gr.Progress()):
    logs = []
    def log(msg): 
        logs.append(msg)
        return "\n".join(logs)

    if not source_input or not model_paths:
        yield ("Error: Missing Input or Model Selection", [])
        return

    if not isinstance(model_paths, list):
        model_paths = [model_paths]

    # Inject Hugging Face Token if provided
    if hf_token_input and hf_token_input.strip():
        os.environ["HF_TOKEN"] = hf_token_input.strip()
        yield (log("🔑 Hugging Face token loaded successfully."), [])

    yield (log(f"--- Fetching metadata from {source_type}: {source_input} ---"), [])
    try:
        if source_type == "Local Folder":
            all_videos = scan_local_folder(source_input)
        elif source_type == "Fichier Vidéo unique (MP4/etc.)":
            import pathlib
            file_path = os.path.abspath(source_input.strip('"').strip())
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Le fichier spécifié n'existe pas : {file_path}")
            filename = pathlib.Path(file_path).stem
            video_info = {
                'id': filename,
                'title': filename,
                'url': file_path
            }
            all_videos = [video_info]
        else:
            urls = [u.strip() for u in source_input.split('\n') if u.strip()]
            all_videos = []
            for url in urls:
                infos = get_video_info(url, proxy=proxy_url)
                all_videos.extend(infos)
        if not all_videos:
            yield (log("❌ No valid videos/files found."), [])
            return
    except Exception as e:
        yield (log(f"❌ Error: {str(e)}"), [])
        return

    all_results_files = []
    batch_size = 50
    total_batches = (len(all_videos) + batch_size - 1) // batch_size

    for b_idx in range(total_batches):
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, len(all_videos))
        batch_videos = all_videos[start_i:end_i]
        
        batch_items = [VideoItem(v) for v in batch_videos]
        if source_type in ["Local Folder", "Fichier Vidéo unique (MP4/etc.)"]:
            for item in batch_items:
                item.wav_path = item.data['url']
                item.status = "Downloaded"
        else:
            batch_items = batch_download_phase(batch_videos, proxy=proxy_url, progress_callback=lambda c, t, m: None)
        
        ready_items = [i for i in batch_items if i.wav_path]
        batch_items = batch_split_phase(batch_items, chunk_length_s, lambda c, t, m: None, overlap_s=overlap_s, split_method=split_method)
        
        split_ready = [i for i in batch_items if i.status == "Split Ready"]

        import pathlib
        subfolder = None
        if source_type == "Local Folder":
            subfolder = f"{pathlib.Path(source_input).name}_transcriptions"
        elif source_type == "Fichier Vidéo unique (MP4/etc.)":
            subfolder = f"{pathlib.Path(source_input).stem}_transcriptions"
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            subfolder = f"url_transcriptions_{timestamp}"

        for m_idx, m_path in enumerate(model_paths):
            for item in split_ready:
                item.status = "Split Ready"
                
            yield (log(f"Batch {b_idx+1}: Transcribing with: {m_path} ({m_idx+1}/{len(model_paths)})"), all_results_files)
            
            devices = get_available_gpus() if device_id == "All Available GPUs" else ([f"cuda:{device_id.split()[-1]}"] if device_id.startswith("GPU ") else ["cpu"])
            if not devices:
                devices = ["cpu"]
            q = queue.Queue()
            for item in split_ready: q.put(item)
            workers = []
            target_devices = devices if device_id == "All Available GPUs" else devices[:1]
            for dev in target_devices: 
                t = GPUTranscriberThread(dev, m_path, q, all_results_files, lambda msg: None, chunk_length_s)
                t.strategy = strategy
                t.overlap_s = overlap_s
                t.subfolder = subfolder
                t.start()
                workers.append(t)
            for w in workers: w.join()
            worker_errors = [w.error for w in workers if w.error]
            if worker_errors:
                yield (log(f"❌ Échec du modèle {m_path}: {worker_errors[0].splitlines()[0]}"), all_results_files)
            else:
                yield (log(f"✅ Modèle terminé : {m_path}"), all_results_files)

        if len(model_paths) > 1:
            from modules.utils import generate_comparison_srt, generate_comparison_txt
            for item in split_ready:
                if len(item.model_results) > 1:
                    model_segments = {m: res.get("segments", []) for m, res in item.model_results.items()}
                    model_texts = {m: res.get("text", "") for m, res in item.model_results.items()}
                    comparison_srt = generate_comparison_srt(model_segments)
                    comparison_txt = generate_comparison_txt(model_texts)
                    base_name = f"{item.data['title']} [{item.data['id']}]_COMPARATIVE"
                    if comparison_srt:
                        p = get_output_path(base_name, "srt", subfolder=subfolder)
                        with open(p, "w", encoding="utf-8") as f: f.write(comparison_srt)
                        all_results_files.append(str(p))
                    if comparison_txt:
                        p = get_output_path(base_name, "txt", subfolder=subfolder)
                        with open(p, "w", encoding="utf-8") as f: f.write(comparison_txt)
                        all_results_files.append(str(p))
        yield (log(f"✅ Batch {b_idx+1} Complete."), all_results_files)

    yield (log("\n🎉 All Done!"), all_results_files)

# =====================================================================
# === 8. Telegram Async Wrapper Pipeline ===
# =====================================================================
async def process_telegram_transcription(channel_name, selected_items, model_paths, strategy, device_id, chunk_length_s, overlap_s, proxy_url, hf_token_input, split_method, progress=gr.Progress()):
    logs = []
    def log(msg): 
        logs.append(msg)
        return "\n".join(logs)

    if not selected_items or not model_paths:
        yield ("Error: Veuillez sélectionner des fichiers et au moins un modèle.", [])
        return

    if not isinstance(model_paths, list):
        model_paths = [model_paths]

    # Inject Hugging Face Token if provided
    if hf_token_input and hf_token_input.strip():
        os.environ["HF_TOKEN"] = hf_token_input.strip()
        yield (log("🔑 Hugging Face token loaded successfully."), [])

    from modules.telegram_scraper import download_telegram_file
    import pathlib
    
    all_videos = []
    yield (log(f"--- Téléchargement de {len(selected_items)} fichiers depuis Telegram ---"), [])
    
    for idx, item in enumerate(selected_items):
        msg_id_str, filename = item.split(":", 1)
        msg_id = int(msg_id_str)
        
        yield (log(f"[{idx+1}/{len(selected_items)}] Téléchargement : {filename}..."), [])
        
        try:
            local_path, was_downloaded = await download_telegram_file(channel_name, msg_id)
            filename_stem = pathlib.Path(local_path).stem
            video_info = {
                'id': filename_stem,
                'title': filename_stem,
                'url': local_path
            }
            all_videos.append(video_info)
            status_msg = "Téléchargé" if was_downloaded else "Déjà présent (ignoré)"
            yield (log(f"✅ {status_msg} : {filename}"), [])
        except Exception as e:
            yield (log(f"❌ Échec du téléchargement de {filename} : {e}"), [])

    if not all_videos:
        yield (log("❌ Aucun fichier n'a pu être téléchargé."), [])
        return

    all_results_files = []
    batch_items = [VideoItem(v) for v in all_videos]
    for item in batch_items:
        item.wav_path = item.data['url']
        item.status = "Downloaded"
        
    yield (log(f"--- Découpage des fichiers audio ({len(batch_items)}) ---"), [])
    batch_items = batch_split_phase(batch_items, chunk_length_s, lambda c, t, m: None, overlap_s=overlap_s, split_method=split_method)
    split_ready = [i for i in batch_items if i.status == "Split Ready"]

    subfolder = f"telegram_{channel_name}_transcriptions"

    for m_idx, m_path in enumerate(model_paths):
        for item in split_ready:
            item.status = "Split Ready"
            
        yield (log(f"Transcribing with: {m_path} ({m_idx+1}/{len(model_paths)})"), all_results_files)
        
        devices = get_available_gpus() if device_id == "All Available GPUs" else ([f"cuda:{device_id.split()[-1]}"] if device_id.startswith("GPU ") else ["cpu"])
        if not devices:
            devices = ["cpu"]
        q = queue.Queue()
        for item in split_ready: q.put(item)
        workers = []
        target_devices = devices if device_id == "All Available GPUs" else devices[:1]
        for dev in target_devices: 
            t = GPUTranscriberThread(dev, m_path, q, all_results_files, lambda msg: None, chunk_length_s)
            t.strategy = strategy
            t.overlap_s = overlap_s
            t.subfolder = subfolder
            t.start()
            workers.append(t)
        for w in workers: w.join()
        worker_errors = [w.error for w in workers if w.error]
        if worker_errors:
            yield (log(f"❌ Échec du modèle {m_path}: {worker_errors[0].splitlines()[0]}"), all_results_files)
        else:
            yield (log(f"✅ Modèle terminé : {m_path}"), all_results_files)

    if len(model_paths) > 1:
        from modules.utils import generate_comparison_srt, generate_comparison_txt
        for item in split_ready:
            if len(item.model_results) > 1:
                model_segments = {m: res.get("segments", []) for m, res in item.model_results.items()}
                model_texts = {m: res.get("text", "") for m, res in item.model_results.items()}
                comparison_srt = generate_comparison_srt(model_segments)
                comparison_txt = generate_comparison_txt(model_texts)
                base_name = f"{item.data['title']} [{item.data['id']}]_COMPARATIVE"
                if comparison_srt:
                    p = get_output_path(base_name, "srt", subfolder=subfolder)
                    with open(p, "w", encoding="utf-8") as f: f.write(comparison_srt)
                    all_results_files.append(str(p))
                if comparison_txt:
                    p = get_output_path(base_name, "txt", subfolder=subfolder)
                    with open(p, "w", encoding="utf-8") as f: f.write(comparison_txt)
                    all_results_files.append(str(p))
                    
    yield (log("\n🎉 All Done!"), all_results_files)

async def process_telegram_download_only(channel_name, selected_items, progress=gr.Progress()):
    logs = []
    def log(msg): 
        logs.append(msg)
        return "\n".join(logs)

    if not selected_items:
        yield ("Error: Veuillez sélectionner des fichiers à télécharger.", [])
        return

    from modules.telegram_scraper import download_telegram_file
    from modules.utils import DOWNLOADS_DIR
    import pathlib
    import os

    # Clean channel name to use as subfolder
    clean_channel = "".join([c for c in channel_name if c.isalnum() or c in ['_', '-']]).strip()
    if not clean_channel:
        clean_channel = "telegram_channel"
        
    dest_dir = os.path.abspath(os.path.join(DOWNLOADS_DIR, f"telegram_{clean_channel}_files"))
    os.makedirs(dest_dir, exist_ok=True)
    
    yield (log(f"--- Début du téléchargement de {len(selected_items)} fichiers ---"), [])
    yield (log(f"Dossier de destination : {dest_dir}\n"), [])
    
    downloaded_paths = []
    for idx, item in enumerate(selected_items):
        msg_id_str, filename = item.split(":", 1)
        msg_id = int(msg_id_str)
        
        yield (log(f"[{idx+1}/{len(selected_items)}] Téléchargement : {filename}..."), downloaded_paths)
        
        try:
            local_path, was_downloaded = await download_telegram_file(channel_name, msg_id, dest_dir=dest_dir)
            downloaded_paths.append(local_path)
            status_msg = "Téléchargé avec succès" if was_downloaded else "Déjà présent (ignoré)"
            yield (log(f"✅ {status_msg} : {filename}"), downloaded_paths)
        except Exception as e:
            yield (log(f"❌ Échec de téléchargement de {filename} : {e}"), downloaded_paths)
            
    yield (log(f"\n🎉 Terminé ! Tous les fichiers ont été placés dans :\n{dest_dir}"), downloaded_paths)

# =====================================================================
# === 9. File Viewer Helper Logic ===
# =====================================================================
def get_output_files():
    files = glob.glob(os.path.join("outputs", "**", "*.srt"), recursive=True) + \
            glob.glob(os.path.join("outputs", "**", "*.txt"), recursive=True)
    files = sorted(files, key=os.path.getmtime, reverse=True)
    return [os.path.relpath(f) for f in files]


def _safe_output_file(filepath):
    if not filepath:
        return None
    outputs_root = os.path.realpath(os.path.join(os.path.dirname(__file__), "outputs"))
    candidate = os.path.realpath(os.path.join(os.path.dirname(__file__), filepath))
    try:
        if os.path.commonpath([outputs_root, candidate]) != outputs_root:
            return None
    except ValueError:
        return None
    return candidate

def read_file_content(filepath):
    if not filepath:
        return "Aucun fichier sélectionné."
    full_path = _safe_output_file(filepath)
    if not full_path:
        return "Erreur : chemin de fichier non autorisé."
    if not os.path.exists(full_path):
        return f"Erreur : Le fichier n'existe plus : {filepath}"
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Erreur : {e}"

def delete_output_file(filepath):
    if not filepath:
        return gr.update(), "Aucun fichier à supprimer."
    full_path = _safe_output_file(filepath)
    if not full_path:
        return gr.update(), "Chemin de fichier non autorisé."
    if os.path.exists(full_path):
        try:
            os.remove(full_path)
            updated_files = get_output_files()
            return gr.update(choices=updated_files, value=updated_files[0] if updated_files else None), "🗑️ Fichier supprimé avec succès."
        except Exception as e:
            return gr.update(), f"Erreur de suppression : {e}"
    return gr.update(), "Le fichier n'existe pas."

# =====================================================================
# === 10. UI Setup ===
# =====================================================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;500;600;700&display=swap');

:root {
    --ink: #17221d;
    --muted: #66736c;
    --paper: #f5f1e8;
    --paper-raised: #fffdf8;
    --green: #173f34;
    --green-bright: #24604f;
    --gold: #b78a3d;
    --line: #ddd4c3;
    --shadow: 0 12px 35px rgba(44, 35, 24, .08);
}

body, .gradio-container {
    color: var(--ink) !important;
    font-family: "Segoe UI", "Noto Sans Arabic", sans-serif !important;
}
body {
    background:
        radial-gradient(circle at 8% 0%, rgba(183, 138, 61, .10), transparent 28rem),
        linear-gradient(180deg, #f8f5ee 0%, #eee7da 100%) !important;
    background-attachment: fixed !important;
}
.gradio-container { width: 100% !important; max-width: none !important; padding: 24px 28px 48px !important; }

.premium-header {
    position: relative;
    overflow: hidden;
    padding: 28px 34px;
    margin-bottom: 18px;
    border: 1px solid rgba(183, 138, 61, .45);
    border-radius: 22px;
    background: linear-gradient(125deg, #102d25 0%, #1b493c 70%, #285b4c 100%);
    box-shadow: 0 18px 45px rgba(23, 63, 52, .22);
    color: #fffdf8;
}
.premium-header::after {
    content: "كتب";
    position: absolute;
    right: 32px;
    top: -36px;
    font: 700 150px/1 "Noto Sans Arabic", sans-serif;
    color: rgba(255,255,255,.045);
}
.premium-header h1 { margin: 0 0 5px !important; font-size: clamp(2rem, 4vw, 3.1rem) !important; letter-spacing: -.04em; }
.premium-header p { margin: 0 !important; max-width: 760px; color: #d7e4dc; font-size: 1.05rem; }
.header-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.header-pills span { padding: 5px 11px; border: 1px solid rgba(255,255,255,.18); border-radius: 999px; background: rgba(255,255,255,.08); color: #f3ead8; font-size: .82rem; }

.tab-nav { gap: 5px !important; padding: 6px !important; border: 1px solid var(--line) !important; border-radius: 14px !important; background: rgba(255,253,248,.82) !important; box-shadow: var(--shadow); }
.tab-nav button { border-radius: 9px !important; font-weight: 650 !important; }
.tab-nav button.selected { color: white !important; background: var(--green) !important; }

.block, .form, .panel, fieldset {
    border-color: var(--line) !important;
    border-radius: 14px !important;
}
.form, .panel { background: rgba(255,253,248,.76) !important; box-shadow: 0 5px 18px rgba(44,35,24,.04); }
.prose h3, .prose h4, .prose h5 { color: var(--green) !important; letter-spacing: -.015em; }
input, textarea { caret-color: var(--green-bright); }

.start-btn, .save-btn, .ocr-save-page-btn {
    border: 0 !important;
    border-radius: 11px !important;
    color: white !important;
    font-weight: 700 !important;
    transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
}
.start-btn { background: linear-gradient(135deg, var(--green) 0%, var(--green-bright) 100%) !important; box-shadow: 0 8px 20px rgba(23,63,52,.20) !important; }
.save-btn { background: linear-gradient(135deg, #2b6655 0%, #367d67 100%) !important; }
.ocr-save-page-btn { background: linear-gradient(135deg, #9b722f 0%, var(--gold) 100%) !important; }
.start-btn:hover, .save-btn:hover, .ocr-save-page-btn:hover { transform: translateY(-1px) !important; filter: brightness(1.06); }

.ocr-guide {
    padding: 13px 16px;
    margin: 8px 0 16px;
    border-left: 4px solid var(--gold);
    border-radius: 0 12px 12px 0;
    background: rgba(255,253,248,.82);
    color: var(--muted);
}
.ocr-file-upload, .ocr-select-dropdown, .ocr-status-box { box-shadow: 0 5px 18px rgba(44,35,24,.06) !important; }
.ocr-textbox textarea {
    direction: rtl !important;
    text-align: right !important;
    min-height: 610px !important;
    border: 1px solid #d8c9ae !important;
    border-radius: 10px !important;
    background: #fffef9 !important;
    color: #171b18 !important;
    font-family: "Noto Sans Arabic", "Segoe UI", sans-serif !important;
    font-size: 1.18rem !important;
    line-height: 2 !important;
}
.ocr-empty { padding: 5rem 2rem; text-align: center; color: var(--muted); border: 1px dashed var(--line); border-radius: 14px; background: rgba(255,253,248,.65); }
.ocr-model-note { color: var(--muted); font-size: .88rem; margin-top: -5px; }
iframe { border: 1px solid var(--line) !important; border-radius: 12px !important; background: white; box-shadow: var(--shadow); }
.ocr-page-image { align-self: flex-start; }
.ocr-page-image img { width: 100% !important; height: 100% !important; object-fit: contain !important; object-position: top center !important; }
.ocr-page-image .image-container { background: #e8e2d7 !important; }

@media (max-width: 900px) {
    .gradio-container { padding: 12px !important; }
    .premium-header { padding: 22px; border-radius: 16px; }
    .premium-header::after { display: none; }
    .ocr-textbox textarea { min-height: 360px !important; }
}
"""

def generate_higgs_ui(text, ref_audio, ref_text, temperature, top_p):
    from modules.higgs_tts import generate_voice_higgs
    if not text or not text.strip():
        return None, "Erreur : Le texte à synthétiser ne peut pas être vide."
    
    output_path, status = generate_voice_higgs(
        text=text,
        ref_audio_path=ref_audio,
        ref_text=ref_text,
        temperature=temperature,
        top_p=top_p
    )
    return output_path, status

def get_pdf_iframe_html(pdf_path):
    import os
    import time
    if not pdf_path or not os.path.exists(pdf_path):
        return "<div style='text-align: center; padding: 2rem; color: #888;'>Aucun PDF chargé</div>"
    try:
        rel_path = os.path.relpath(pdf_path, os.getcwd()).replace("\\", "/")
        iframe_src = f"/gradio_api/file={rel_path}?t={int(time.time())}"
    except Exception:
        abs_path = os.path.abspath(pdf_path).replace("\\", "/")
        iframe_src = f"/gradio_api/file={abs_path}?t={int(time.time())}"
    return f'<iframe src="{iframe_src}" width="100%" height="800px" style="border: 2px solid #162447; border-radius: 12px; background: white;"></iframe>'

def get_ocr_page_image(pdf_path, page_num):
    """Return the rendered PDF page used by OCR so both panes stay aligned."""
    if not pdf_path or not page_num:
        return None
    image_path = os.path.join(
        os.path.dirname(os.path.abspath(pdf_path)),
        "images",
        f"page_{int(page_num):04d}.png",
    )
    return image_path if os.path.exists(image_path) else None


def _build_batches(total_pages, chunk=15):
    return [f"Pages {i+1} à {min(i+chunk, total_pages)}"
            for i in range(0, total_pages, chunk)]

def _empty_pdf_ui(status):
    from modules.qari_ocr import get_cached_pdfs
    return [
        None,
        gr.update(choices=[], value=None),
        gr.update(choices=get_cached_pdfs(), value=None),
        None,
        status,
        None,
        gr.update(value="", label="Qari — PAGE —"),
        gr.update(choices=[], value=None),
    ]


def _loaded_pdf_ui(pdf_path, total_pages, third_component):
    from modules.qari_ocr import (
        get_cached_pdfs,
        get_ocr_text_path as qari_path,
        get_page_text as qari_page,
    )

    batches = _build_batches(total_pages)
    pages = [str(page) for page in range(1, total_pages + 1)]
    first_page = 1 if pages else None
    return [
        pdf_path,
        gr.update(choices=batches, value=batches[0] if batches else None),
        third_component,
        qari_path(pdf_path) if os.path.exists(qari_path(pdf_path)) else None,
        f"✅ {os.path.basename(pdf_path)} prêt — {total_pages} pages, images OCR à 300 DPI.",
        gr.update(
            value=get_ocr_page_image(pdf_path, first_page),
            label=f"Page PDF — PAGE {first_page}" if first_page else "Page PDF — PAGE —",
        ),
        gr.update(
            value=qari_page(pdf_path, first_page) if first_page else "",
            label=f"Qari — PAGE {first_page}" if first_page else "Qari — PAGE —",
        ),
        gr.update(choices=pages, value=pages[0] if pages else None),
    ]

def handle_pdf_clear():
    return _empty_pdf_ui("Fichier effacé.")

def handle_pdf_upload_change(file_obj, progress=gr.Progress()):
    from modules.qari_ocr import prepare_pdf_cache, get_cached_pdfs
    if not file_obj:
        return handle_pdf_clear()

    pdf_path = file_obj if isinstance(file_obj, str) else (file_obj.name if hasattr(file_obj, 'name') else (file_obj.get('name') if isinstance(file_obj, dict) else None))
    if not pdf_path or not os.path.exists(pdf_path):
        return _empty_pdf_ui("Fichier introuvable.")

    pdf_path, total_pages = prepare_pdf_cache(pdf_path, progress)
    if not pdf_path:
        return _empty_pdf_ui("Erreur pendant la préparation du PDF.")
    cached_update = gr.update(choices=get_cached_pdfs(), value=os.path.basename(pdf_path))
    return _loaded_pdf_ui(pdf_path, total_pages, cached_update)

def handle_pdf_dropdown_change(cached_dropdown, progress=gr.Progress()):
    from modules.qari_ocr import prepare_pdf_cache, get_pdf_workspace_dir
    if not cached_dropdown:
        result = _empty_pdf_ui("Aucun PDF sélectionné.")
        result[2] = gr.update(value=None)
        return result

    pdf_path = os.path.join(get_pdf_workspace_dir(cached_dropdown), cached_dropdown)
    if not pdf_path or not os.path.exists(pdf_path):
        result = _empty_pdf_ui("Fichier introuvable.")
        result[2] = gr.update(value=None)
        return result

    pdf_path, total_pages = prepare_pdf_cache(pdf_path, progress)
    if not pdf_path:
        result = _empty_pdf_ui("Erreur pendant la préparation du PDF.")
        result[2] = gr.update(value=None)
        return result
    return _loaded_pdf_ui(pdf_path, total_pages, gr.update(value=None))

def handle_save_full_text(pdf_path, text_content):
    from modules.qari_ocr import get_ocr_text_path
    if not pdf_path:
        return "Erreur : Aucun PDF chargé."
    ocr_path = get_ocr_text_path(pdf_path)
    os.makedirs(os.path.dirname(ocr_path), exist_ok=True)
    with open(ocr_path, "w", encoding="utf-8") as f:
        f.write(text_content)
    return "✅ Transcription enregistrée !"

def handle_page_change(pdf_path, page_num):
    """Synchronize the rendered PDF page with its Qari transcription."""
    if not pdf_path or not page_num:
        return None, gr.update(value="[Non transcrit]")
    from modules.qari_ocr import get_page_text as qari_page
    page = int(page_num)
    return (
        gr.update(
            value=get_ocr_page_image(pdf_path, page),
            label=f"Page PDF — PAGE {page}",
        ),
        gr.update(value=qari_page(pdf_path, page), label=f"Qari — PAGE {page}"),
    )


def handle_page_step(pdf_path, page_num, delta):
    """Move one page while keeping the image and both texts synchronized."""
    if not pdf_path:
        image, qari_text = handle_page_change(None, None)
        return gr.update(), image, qari_text
    from modules.qari_ocr import get_pdf_total_pages
    total = get_pdf_total_pages(pdf_path)
    current = int(page_num or 1)
    target = min(max(current + int(delta), 1), max(total, 1))
    image, qari_text = handle_page_change(pdf_path, target)
    return str(target), image, qari_text

def run_ocr_generator(pdf_path, batch_str, model_choice, all_pdf=False, current_page=None):

    """
    Streaming generator for Qari OCR.
    Yields: (live_stream, dl_qari, status, full_qari)
    """
    import threading, queue as q_mod

    if not pdf_path:
        yield gr.update(visible=False), None, "Erreur : Aucun PDF chargé.", ""
        return

    if all_pdf:
        from modules.qari_ocr import get_pdf_total_pages
        total = get_pdf_total_pages(pdf_path)
        batches = [(i+1, min(i+15, total)) for i in range(0, total, 15)]
    else:
        try:
            import re
            bounds = [int(value) for value in re.findall(r"\d+", batch_str or "")]
            if len(bounds) != 2:
                raise ValueError
            batches = [(bounds[0], bounds[1])]
        except (TypeError, ValueError):
            yield gr.update(visible=False), None, "Erreur sélection lot.", ""
            return

    q = q_mod.Queue()
    error_box = [None]

    def thread_fn():
        try:
            import modules.qari_ocr as qari_mod

            for s, e in batches:
                q.put(f"\n🔍 Qari 0.4.0 — pages {s}–{e}\n")
                result = qari_mod.run_ocr_batch(pdf_path, s, e, q.put)
                q.put(f"{result}\n")
                if result.startswith("Échec") or result.startswith("Erreur"):
                    raise RuntimeError(result)

            import torch, gc; torch.cuda.empty_cache(); gc.collect()
        except Exception as exc:
            error_box[0] = exc
            import traceback
            q.put(f"\n❌ ERREUR: {exc}\n{traceback.format_exc()}\n")

    t = threading.Thread(target=thread_fn, daemon=True)
    t.start()

    stream = ""
    yield gr.update(value="", visible=True), gr.update(), "Démarrage OCR…", gr.update()

    while t.is_alive() or not q.empty():
        try:
            chunk = q.get(timeout=0.2)
            stream += chunk
            yield gr.update(value=stream, visible=True), gr.update(), "OCR en cours…", gr.update()
        except q_mod.Empty:
            continue

    from modules.qari_ocr import get_ocr_text_path as qari_path
    from modules.qari_ocr import get_page_text as qari_page_text
    selected_page = int(current_page or batches[0][0])
    page_r = qari_page_text(pdf_path, selected_page)
    dl_r = qari_path(pdf_path) if os.path.exists(qari_path(pdf_path)) else None
    status = "✅ OCR terminé et sauvegardé page par page." if not error_box[0] else f"❌ {error_box[0]}"
    yield gr.update(visible=False), dl_r, status, page_r


with gr.Blocks(
    title="Al-Kutub — Atelier de transcription",
    theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="stone"),
    css=custom_css,
    fill_width=True,
) as demo:
    gr.HTML("""
    <div class="premium-header">
        <h1>Al-Kutub</h1>
        <p>Un atelier local pour transcrire la parole, numériser les livres arabes et comparer les modèles.</p>
        <div class="header-pills"><span>OCR arabe</span><span>Audio & vidéo</span><span>Traitement local</span><span>Reprise automatique</span></div>
    </div>
    """)
    
    with gr.Tabs():
        # --- TAB 1: Direct Transcription ---
        with gr.Tab("🎙️ Transcrire"):
            with gr.Row():
                with gr.Column(scale=6):
                    with gr.Group():
                        source_type = gr.Radio(
                            choices=[
                                ("Liens web (YouTube / Facebook)", "URL List (YouTube/Facebook)"),
                                ("Dossier local", "Local Folder"),
                                ("Fichier audio ou vidéo", "Fichier Vidéo unique (MP4/etc.)"),
                            ],
                            label="📂 Type de source",
                            value="URL List (YouTube/Facebook)"
                        )
                        source_input = gr.Textbox(
                            label="🔗 Liste d'URLs (Une par ligne)",
                            placeholder="https://youtube.com/watch?v=...",
                            lines=5
                        )
                    
                    with gr.Group():
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                choices=list_available_models(),
                                label="🤖 Modèles de transcription",
                                value=[list_available_models()[0]] if list_available_models() else [],
                                multiselect=True
                            )
                            strategy_radio = gr.Radio(
                                choices=[("Rapide (CTC)", "ctc_greedy"), ("Standard (RNNT)", "rnnt_greedy"), ("Précis (RNNT Beam)", "rnnt_beam")],
                                label="⚙️ Stratégie",
                                value="rnnt_greedy"
                            )
                        
                        with gr.Row():
                            hf_token_input = gr.Textbox(
                                label="🔑 Token Hugging Face",
                                placeholder="hf_...",
                                type="password",
                                value=env_token
                            )
                    
                    with gr.Accordion("🛠️ Paramètres avancés & Performance", open=False):
                        with gr.Row():
                            device_dropdown = gr.Dropdown(
                                choices=get_device_choices(),
                                label="💻 Calcul",
                                value="GPU 0" if torch.cuda.is_available() else "CPU"
                            )
                            split_strategy = gr.Radio(
                                choices=["VAD (silence intelligent)", "Découpage fixe (secondes)"],
                                value="Découpage fixe (secondes)",
                                label="✂️ Méthode de découpage audio (Recommandé: Découpage fixe)"
                            )
                        with gr.Row():
                            chunk_slider = gr.Slider(
                                label="⏱️ Longueur des segments (s) [Découpage fixe]",
                                minimum=3,
                                maximum=600,
                                value=30,
                                step=1
                            )
                            overlap_slider = gr.Slider(
                                label="🔄 Chevauchement (Overlap s) [Découpage fixe]",
                                minimum=0,
                                maximum=30,
                                value=5,
                                step=1
                            )
                        proxy_input = gr.Textbox(
                            label="🌐 URL Proxy (Optionnel)",
                            placeholder="http://127.0.0.1:7890"
                        )
                    
                    run_btn = gr.Button("🚀 Démarrer la transcription", variant="primary", elem_classes=["start-btn"])
                    
                with gr.Column(scale=4):
                    logs_output = gr.Textbox(
                        label="📝 Historique & Logs",
                        lines=12,
                        interactive=False
                    )
                    files_output = gr.File(
                        label="📥 Fichiers de sortie (SRT / TXT)",
                        file_count="multiple"
                    )
        
        # --- TAB 2: Telegram Scraper ---
        with gr.Tab("✈️ Telegram"):
            with gr.Row():
                with gr.Column(scale=5):
                    # Step 2a: Authentication Card
                    with gr.Group():
                        gr.Markdown("### 🔑 Connexion Telegram (Utilisateur)")
                        with gr.Row():
                            tg_api_id_input = gr.Textbox(label="API ID", value=env_api_id, placeholder="Ex: 1234567")
                            tg_api_hash_input = gr.Textbox(label="API Hash", value=env_api_hash, placeholder="Ex: abcdef0123...", type="password")
                        with gr.Row():
                            tg_phone_input = gr.Textbox(label="Numéro de Téléphone", value=env_phone, placeholder="Ex: +33612345678")
                            send_code_btn = gr.Button("✉️ Envoyer le code", variant="secondary")
                        
                        # SMS Code & Connect (Hidden initially)
                        tg_code_input = gr.Textbox(label="Code de vérification", placeholder="Ex: 12345", visible=False)
                        connect_btn = gr.Button("✅ Se connecter", variant="primary", visible=False, elem_classes=["save-btn"])
                        
                        auth_status_output = gr.Markdown("*Non connecté*")
                        
                        # State managers
                        phone_code_hash_state = gr.State()
                        telegram_files_choices_state = gr.State(value=[])
                        telegram_new_files_state = gr.State(value=[])
                    
                    # Step 2b: Scraping Card
                    with gr.Group():
                        gr.Markdown("### 📡 Recherche de fichiers multimédias")
                        with gr.Row():
                            tg_channel_input = gr.Textbox(label="Canal / Groupe Telegram", value="httpstmetUEqBLBc9tQ2MWE0", placeholder="Ex: NomDuCanal")
                            with gr.Column():
                                tg_limit_slider = gr.Slider(minimum=10, maximum=5000, step=50, value=500, label="Messages à parcourir")
                                tg_all_history_checkbox = gr.Checkbox(label="Parcourir tout l'historique (Plus long)", value=False)
                        fetch_files_btn = gr.Button("🔍 Lister les fichiers", variant="secondary")
                        
                        telegram_files_checkboxes = gr.CheckboxGroup(choices=[], label="Fichiers détectés dans le groupe")
                        with gr.Row():
                            tg_select_all_btn = gr.Button("✅ Tout sélectionner", variant="secondary")
                            tg_select_new_btn = gr.Button("🆕 Sélectionner nouveaux", variant="secondary")
                            tg_deselect_all_btn = gr.Button("❌ Tout désélectionner", variant="secondary")
                        
                        with gr.Row():
                            tg_download_only_btn = gr.Button("📥 Télécharger uniquement", variant="secondary")
                            tg_transcribe_btn = gr.Button("🚀 Télécharger & Transcrire sélectionnés", variant="primary", elem_classes=["start-btn"])
                        
                with gr.Column(scale=5):
                    tg_logs_output = gr.Textbox(
                        label="📝 Logs Telegram",
                        lines=15,
                        interactive=False
                    )
                    tg_files_output = gr.File(
                        label="📥 Fichiers générés",
                        file_count="multiple"
                    )

        # --- TAB 3: File Visualizer ---
        with gr.Tab("📁 Fichiers"):
            with gr.Row():
                with gr.Column(scale=4):
                    file_list_dropdown = gr.Dropdown(
                        choices=get_output_files(),
                        label="📄 Transcriptions générées dans outputs/",
                        value=get_output_files()[0] if get_output_files() else None
                    )
                    with gr.Row():
                        refresh_files_btn = gr.Button("🔄 Rafraîchir")
                        delete_file_btn = gr.Button("🗑️ Supprimer le fichier", variant="stop")
                    file_info_output = gr.Markdown()
                with gr.Column(scale=6):
                    file_preview_code = gr.Code(
                        label="📝 Aperçu du contenu",
                        language="markdown",
                        interactive=False,
                        lines=22
                    )

        # --- TAB 3.5: Higgs TTS Synthesizer ---
        with gr.Tab("🔊 Voix"):
            gr.Markdown("### 🔊 Synthèse de voix expressive et clonage de voix via Higgs TTS 3 (4B)")
            with gr.Row():
                with gr.Column(scale=5):
                    higgs_text = gr.Textbox(
                        label="✍️ Texte à synthétiser (Français, Anglais ou Arabe)",
                        placeholder="أدخل النص هنا للتوليد الصوتي...",
                        lines=5
                    )
                    with gr.Row():
                        higgs_temp = gr.Slider(
                            label="🎭 Expressivité (Temperature)",
                            minimum=0.1,
                            maximum=1.2,
                            value=0.7,
                            step=0.05
                        )
                        higgs_top_p = gr.Slider(
                            label="🎯 Précision (Top-P)",
                            minimum=0.1,
                            maximum=1.0,
                            value=0.95,
                            step=0.05
                        )
                with gr.Column(scale=5):
                    higgs_ref_audio = gr.Audio(
                        label="🎙️ Audio de Référence pour le Clonage (Optionnel)",
                        type="filepath"
                    )
                    higgs_ref_text = gr.Textbox(
                        label="📝 Transcription de l'audio de référence (Optionnel mais recommandé)",
                        placeholder="Transcription exacte du clip audio de référence..."
                    )
            with gr.Row():
                higgs_btn = gr.Button("🔊 Générer la voix", variant="primary", elem_classes=["save-btn"])
            
            with gr.Row():
                higgs_audio_output = gr.Audio(label="🎵 Voix Synthétisée")
                higgs_status_output = gr.Textbox(label="📊 Statut / Logs", interactive=False)

            higgs_btn.click(
                generate_higgs_ui,
                inputs=[higgs_text, higgs_ref_audio, higgs_ref_text, higgs_temp, higgs_top_p],
                outputs=[higgs_audio_output, higgs_status_output]
            )

        # --- TAB 3.7: OCR de Documents (Qari) ---
        with gr.Tab("📖 OCR arabe"):
            from modules.qari_ocr import get_cached_pdfs

            gr.Markdown("### Numérisation de livres et manuscrits arabes")
            gr.HTML("""
            <div class="ocr-guide"><strong>Qari OCR 0.4.0 — moteur Unsloth, quantification 8 bits</strong>
            privilégie la fidélité sur les livres arabes imprimés. Le modèle est chargé en 8 bits
            pour préserver la précision des petits caractères. Son architecture contient 4 milliards de paramètres.
            Une page 300 DPI peut prendre plusieurs minutes.</div>
            """)
            active_pdf_path = gr.State()

            with gr.Row():
                with gr.Column(scale=3):
                    ocr_pdf_file = gr.File(
                        label="📂 Glissez-déposez votre PDF",
                        file_types=[".pdf"],
                        elem_classes=["ocr-file-upload"]
                    )
                    ocr_cached_dropdown = gr.Dropdown(
                        choices=get_cached_pdfs(),
                        label="📄 Ou sélectionnez un PDF déjà traité",
                        value=None,
                        elem_classes=["ocr-select-dropdown"]
                    )
                    ocr_refresh_pdfs_btn = gr.Button("🔄 Rafraîchir la liste", elem_classes=["ocr-refresh-btn"])

                with gr.Column(scale=5):
                    gr.Markdown("#### 🔍 Qari OCR 0.4.0 — 8 bits")
                    ocr_model_choice = gr.State("Qari OCR 0.4.0")
                    ocr_batch_dropdown = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="📦 Lot à transcrire (Max 15 pages par lot)"
                    )
                    with gr.Row():
                        ocr_run_batch_btn = gr.Button("📄 Transcrire ce lot", variant="primary", elem_classes=["save-btn"])
                        ocr_run_all_btn   = gr.Button("🚀 Transcrire TOUT le PDF", variant="primary", elem_classes=["start-btn"])

                with gr.Column(scale=4):
                    ocr_status = gr.Textbox(label="📊 Statut", lines=3, interactive=False, elem_classes=["ocr-status-box"])
                    ocr_dl_qari = gr.File(label="⬇️ Qari 0.4.0 (.txt)", interactive=False)

            ocr_live_stream = gr.Textbox(
                label="🚀 Flux Live (progression page par page)",
                lines=8, max_lines=12,
                interactive=False, visible=False,
                elem_classes=["ocr-status-box"]
            )

            gr.Markdown("---\n#### Lecture et correction page par page")

            with gr.Row():
                ocr_page_prev_btn = gr.Button("← Page précédente", scale=1)
                ocr_page_selector = gr.Dropdown(
                    choices=[],
                    label="🔢 Sélectionner la page",
                    scale=3
                )
                ocr_page_load_btn = gr.Button("📖 Charger cette page", scale=1, elem_classes=["save-btn"])
                ocr_page_next_btn = gr.Button("Page suivante →", scale=1)

            with gr.Row(equal_height=False):
                with gr.Column(scale=6, min_width=420):
                    gr.Markdown("##### 🖼️ Page du PDF")
                    ocr_pdf_viewer = gr.Image(
                        value=None,
                        label="Page PDF — PAGE —",
                        type="filepath",
                        interactive=False,
                        height=1100,
                        buttons=["fullscreen"],
                        elem_classes=["ocr-page-image"],
                    )

                with gr.Column(scale=6, min_width=360):
                    gr.Markdown("##### 🔍 Qari OCR 0.4.0")
                    ocr_qari_text = gr.Textbox(
                        label="Transcription Qari",
                        lines=30, max_lines=50,
                        interactive=True,
                        elem_classes=["ocr-textbox"],
                        rtl=True
                    )
                    save_qari_btn = gr.Button("💾 Sauvegarder Qari", variant="secondary", elem_classes=["ocr-save-page-btn"])

            _pdf_load_outputs = [
                active_pdf_path,
                ocr_batch_dropdown,
                ocr_cached_dropdown,
                ocr_dl_qari,
                ocr_status,
                ocr_pdf_viewer,
                ocr_qari_text,
                ocr_page_selector,
            ]

            ocr_refresh_pdfs_btn.click(
                lambda: gr.update(choices=get_cached_pdfs()),
                outputs=[ocr_cached_dropdown]
            )
            ocr_pdf_file.upload(
                handle_pdf_upload_change,
                inputs=[ocr_pdf_file],
                outputs=_pdf_load_outputs
            )
            ocr_pdf_file.clear(
                handle_pdf_clear,
                outputs=_pdf_load_outputs
            )
            ocr_cached_dropdown.input(
                handle_pdf_dropdown_change,
                inputs=[ocr_cached_dropdown],
                outputs=[
                    active_pdf_path,
                    ocr_batch_dropdown,
                    ocr_pdf_file,
                    ocr_dl_qari,
                    ocr_status,
                    ocr_pdf_viewer,
                    ocr_qari_text,
                    ocr_page_selector,
                ]
            )
            ocr_page_load_btn.click(
                handle_page_change,
                inputs=[active_pdf_path, ocr_page_selector],
                outputs=[ocr_pdf_viewer, ocr_qari_text]
            )
            ocr_page_selector.input(
                handle_page_change,
                inputs=[active_pdf_path, ocr_page_selector],
                outputs=[ocr_pdf_viewer, ocr_qari_text]
            )
            ocr_page_prev_btn.click(
                lambda pdf_path, page_num: handle_page_step(pdf_path, page_num, -1),
                inputs=[active_pdf_path, ocr_page_selector],
                outputs=[ocr_page_selector, ocr_pdf_viewer, ocr_qari_text],
            )
            ocr_page_next_btn.click(
                lambda pdf_path, page_num: handle_page_step(pdf_path, page_num, 1),
                inputs=[active_pdf_path, ocr_page_selector],
                outputs=[ocr_page_selector, ocr_pdf_viewer, ocr_qari_text],
            )

            def save_qari_page(pdf_path, page_num, text):
                if not pdf_path or not page_num:
                    return "Erreur : PDF ou page non définis."
                from modules.qari_ocr import load_parsed_ocr, save_parsed_ocr
                d = load_parsed_ocr(pdf_path)
                d[int(page_num)] = text
                save_parsed_ocr(pdf_path, d)
                return f"✅ Page {page_num} Qari sauvegardée."

            save_qari_btn.click(
                save_qari_page,
                inputs=[active_pdf_path, ocr_page_selector, ocr_qari_text],
                outputs=[ocr_status]
            )
            ocr_run_batch_btn.click(
                fn=run_ocr_generator,
                inputs=[active_pdf_path, ocr_batch_dropdown, ocr_model_choice, gr.State(value=False), ocr_page_selector],
                outputs=[ocr_live_stream, ocr_dl_qari, ocr_status, ocr_qari_text]
            )
            ocr_run_all_btn.click(
                fn=run_ocr_generator,
                inputs=[active_pdf_path, ocr_batch_dropdown, ocr_model_choice, gr.State(value=True), ocr_page_selector],
                outputs=[ocr_live_stream, ocr_dl_qari, ocr_status, ocr_qari_text]
            )


        # --- TAB 4: Configuration Settings ---
        with gr.Tab("⚙️ Réglages"):
            with gr.Group():
                gr.Markdown("### ⚙️ Enregistrer vos informations d'identification dans le fichier `.env`")
                conf_hf_token = gr.Textbox(label="Token Hugging Face (HF_TOKEN)", value=env_token, type="password")
                conf_api_id = gr.Textbox(label="Telegram API ID (TG_API_ID)", value=env_api_id)
                conf_api_hash = gr.Textbox(label="Telegram API Hash (TG_API_HASH)", value=env_api_hash, type="password")
                conf_phone = gr.Textbox(label="Telegram Téléphone (TG_PHONE)", value=env_phone)
                
                save_conf_btn = gr.Button("💾 Enregistrer la configuration", variant="primary", elem_classes=["save-btn"])
                conf_status_output = gr.Markdown()

    # --- Actions: Configuration Settings ---
    save_conf_btn.click(
        save_env_values,
        inputs=[conf_hf_token, conf_api_id, conf_api_hash, conf_phone],
        outputs=[conf_status_output]
    )

    # --- Actions: Tab 1 (Direct Transcription) ---
    def update_source_input(source):
        if source == "Local Folder":
            return gr.update(label="📁 Chemin du dossier local", placeholder="Ex: C:/Users/Downloads/MyVideos", value="")
        elif source == "Fichier Vidéo unique (MP4/etc.)":
            return gr.update(label="🎬 Chemin du fichier vidéo (MP4/MKV/etc.)", placeholder="Ex: C:/Users/Downloads/video.mp4", value="")
        else:
            return gr.update(label="🔗 Liste d'URLs (Une par ligne)", placeholder="https://youtube.com/watch?v=...\nhttps://facebook.com/...", value="")

    source_type.change(update_source_input, inputs=[source_type], outputs=[source_input])
    
    run_btn.click(
        process_content_batch,
        inputs=[source_type, source_input, model_dropdown, strategy_radio, device_dropdown, chunk_slider, overlap_slider, proxy_input, hf_token_input, split_strategy],
        outputs=[logs_output, files_output]
    )

    # --- Actions: Tab 2 (Telegram Scraper) ---
    async def send_code_handler(api_id, api_hash, phone):
        if not api_id or not api_hash or not phone:
            return gr.update(), gr.update(), None, "⚠️ API ID, API Hash et Téléphone requis."
        try:
            try:
                save_env_values(None, None, None, phone)
            except Exception:
                pass
            from modules.telegram_scraper import is_telegram_authorized, send_telegram_code
            if await is_telegram_authorized(api_id, api_hash):
                return gr.update(visible=False), gr.update(visible=False), None, "🟢 Déjà connecté à Telegram ! Vous pouvez lister les fichiers."
            code_hash = await send_telegram_code(api_id, api_hash, phone)
            return gr.update(visible=True), gr.update(visible=True), code_hash, "✅ Code envoyé sur Telegram !"
        except Exception as e:
            return gr.update(), gr.update(), None, f"❌ Erreur : {e}"

    send_code_btn.click(
        send_code_handler,
        inputs=[tg_api_id_input, tg_api_hash_input, tg_phone_input],
        outputs=[tg_code_input, connect_btn, phone_code_hash_state, auth_status_output]
    )

    async def sign_in_handler(phone, code_hash, code):
        if not code or not code_hash:
            return "⚠️ Code ou Hash requis."
        try:
            from modules.telegram_scraper import sign_in_telegram
            success, msg = await sign_in_telegram(phone, code_hash, code)
            if success:
                return "🎉 Connexion réussie et autorisée !"
            else:
                return f"❌ Échec : {msg}"
        except Exception as e:
            return f"❌ Erreur : {e}"

    connect_btn.click(
        sign_in_handler,
        inputs=[tg_phone_input, phone_code_hash_state, tg_code_input],
        outputs=[auth_status_output]
    )

    async def fetch_telegram_files_handler(channel_name, limit, all_history):
        if not channel_name:
            return gr.update(choices=[]), "⚠️ Canal ou Groupe requis.", [], []
        try:
            from modules.telegram_scraper import list_channel_files
            scan_limit = None if all_history else limit
            files = await list_channel_files(channel_name, scan_limit)
            if not files:
                return gr.update(choices=[]), "ℹ️ Aucun fichier audio ou vidéo trouvé dans ce canal.", [], []
            
            choices = []
            choices_values = []
            new_choices_values = []
            for f in files:
                val = f"{f['msg_id']}:{f['name']}"
                choices_values.append(val)
                
                if f.get("exists", False):
                    label = f"✔️ [Déjà présent] {f['name']} ({f['size']}) — {f['date']}"
                else:
                    label = f"🆕 {f['name']} ({f['size']}) — {f['date']}"
                    new_choices_values.append(val)
                    
                choices.append((label, val))
                
            status_text = f"✅ {len(files)} fichiers trouvés ({len(new_choices_values)} nouveaux, {len(files) - len(new_choices_values)} déjà présents) !"
            return gr.update(choices=choices, value=new_choices_values), status_text, choices_values, new_choices_values
        except Exception as e:
            return gr.update(choices=[]), f"❌ Erreur : {e}", [], []

    fetch_files_btn.click(
        fetch_telegram_files_handler,
        inputs=[tg_channel_input, tg_limit_slider, tg_all_history_checkbox],
        outputs=[telegram_files_checkboxes, auth_status_output, telegram_files_choices_state, telegram_new_files_state]
    )

    def select_all_files(choices):
        return gr.update(value=choices)

    def select_new_files(new_choices):
        return gr.update(value=new_choices)

    def deselect_all_files():
        return gr.update(value=[])

    tg_select_all_btn.click(
        select_all_files,
        inputs=[telegram_files_choices_state],
        outputs=[telegram_files_checkboxes]
    )

    tg_select_new_btn.click(
        select_new_files,
        inputs=[telegram_new_files_state],
        outputs=[telegram_files_checkboxes]
    )

    tg_deselect_all_btn.click(
        deselect_all_files,
        outputs=[telegram_files_checkboxes]
    )

    tg_transcribe_btn.click(
        process_telegram_transcription,
        inputs=[tg_channel_input, telegram_files_checkboxes, model_dropdown, strategy_radio, device_dropdown, chunk_slider, overlap_slider, proxy_input, hf_token_input, split_strategy],
        outputs=[tg_logs_output, tg_files_output]
    )

    tg_download_only_btn.click(
        process_telegram_download_only,
        inputs=[tg_channel_input, telegram_files_checkboxes],
        outputs=[tg_logs_output, tg_files_output]
    )

    # --- Actions: Tab 3 (File Visualizer) ---
    def load_selected_file(filepath):
        content = read_file_content(filepath)
        return content, f"📄 **Fichier :** `{filepath}`"

    file_list_dropdown.change(
        load_selected_file,
        inputs=[file_list_dropdown],
        outputs=[file_preview_code, file_info_output]
    )

    def refresh_file_list():
        files = get_output_files()
        val = files[0] if files else None
        return gr.update(choices=files, value=val)

    refresh_files_btn.click(
        refresh_file_list,
        outputs=[file_list_dropdown]
    )

    delete_file_btn.click(
        delete_output_file,
        inputs=[file_list_dropdown],
        outputs=[file_list_dropdown, file_preview_code]
    )

    async def check_telegram_status():
        try:
            from modules.telegram_scraper import is_telegram_authorized
            if await is_telegram_authorized():
                return "🟢 Connecté à Telegram ! Prêt à l'action."
            else:
                return "❌ Non connecté (En attente d'authentification)"
        except Exception as e:
            return f"⚠️ Erreur de connexion automatique : {e}"

    demo.load(
        check_telegram_status,
        outputs=[auth_status_output]
    )

if __name__ == "__main__":
    import socket

    requested_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    server_port = requested_port
    for candidate in range(requested_port, requested_port + 11):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
                server_port = candidate
                break
            except OSError:
                continue
    else:
        raise OSError(f"Aucun port disponible entre {requested_port} et {requested_port + 10}.")

    if server_port != requested_port:
        logger.warning("Port %s occupé ; démarrage sur %s.", requested_port, server_port)

    allowed_dir = os.path.abspath("outputs")
    allowed_paths = [allowed_dir]
    if len(allowed_dir) > 2 and allowed_dir[1] == ':':
        drive = allowed_dir[0]
        rest = allowed_dir[1:]
        allowed_paths.append(drive.upper() + rest)
        allowed_paths.append(drive.lower() + rest)
        
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=server_port,
        inbrowser=os.getenv("GRADIO_INBROWSER", "0") == "1",
        allowed_paths=allowed_paths
    )

