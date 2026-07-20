import os
import sys
import logging
import signal
from pathlib import Path

os.environ['HF_HOME'] = r'J:\UnivHm_NextJS\Al-Kutub-Automator\.cache\huggingface'

# --- CRITICAL: Windows Patch for NeMo/Pytorch Distributed ---
# Windows does not have signal.SIGKILL, but some libraries (like NeMo/PyTorch Lightning) 
# try to access it. We monkeypatch it to SIGTERM or 9 to prevent crashes.
if sys.platform == "win32":
    if not hasattr(signal, "SIGKILL"):
        signal.SIGKILL = 9  # Standard SIGKILL value
        print("INFO: Patched signal.SIGKILL for Windows compatibility.")

# --- CRITICAL: Windows File Lock Patches ---
import shutil
if os.name == 'nt':
    _original_remove = os.remove
    _original_unlink = os.unlink
    _original_rmtree = shutil.rmtree
    
    def _safe_remove(path):
        try:
            _original_remove(path)
        except (PermissionError, OSError):
            pass  # Ignore file locking errors on Windows
    
    def _safe_unlink(path):
        try:
            _original_unlink(path)
        except (PermissionError, OSError):
            pass
    
    def _safe_rmtree(path, *args, **kwargs):
        kwargs['ignore_errors'] = True
        try:
            _original_rmtree(path, *args, **kwargs)
        except (PermissionError, OSError):
            pass  # Ignore "directory not empty" errors
    
    os.remove = _safe_remove
    os.unlink = _safe_unlink
    shutil.rmtree = _safe_rmtree

# --- Logging Setup ---
def setup_logging(name="Al-Kutub", log_file="app.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

logger = setup_logging()

# --- Path Management ---
BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for d in [DOWNLOADS_DIR, OUTPUTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def sanitize_path_prevent_dotfile(filepath):
    """
    If the file name starts with a dot '.', rename it to prevent Gradio exceptions.
    If the destination file already exists, delete the source dotfile and return new_path.
    """
    if not filepath:
        return filepath
    filepath = str(filepath)
    dir_name = os.path.dirname(filepath)
    base_name = os.path.basename(filepath)
    if base_name.startswith('.'):
        new_base_name = "file_" + base_name.lstrip('.')
        if not new_base_name or new_base_name == "file_":
            new_base_name = "file_unnamed"
        new_path = os.path.join(dir_name, new_base_name)
        try:
            if os.path.exists(filepath):
                if os.path.exists(new_path):
                    try:
                        os.remove(filepath)
                    except OSError:
                        pass
                else:
                    os.rename(filepath, new_path)
            return new_path
        except OSError:
            return filepath
    return filepath

def get_output_path(filename, extension="txt", subfolder=None):
    """Returns a safe path in the outputs directory, optionally inside a subfolder."""
    clean_name = "".join(x for x in filename if x.isalnum() or x in "._- ")
    clean_name = clean_name.lstrip('.')
    if not clean_name:
        clean_name = "output"
    target_dir = OUTPUTS_DIR
    if subfolder:
        target_dir = OUTPUTS_DIR / subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{clean_name}.{extension}"

def generate_comparison_srt(model_results, time_bin_size=8.0):
    """
    Combines segments from multiple models into a single comparative SRT.
    model_results: dict mapping model_name to list of segments: {"start": float, "end": float, "text": str}
    """
    # 1. Find the maximum duration from all segments
    max_time = 0.0
    for model_name, segments in model_results.items():
        if segments and isinstance(segments, list):
            for seg in segments:
                if isinstance(seg, dict) and seg.get("end", 0.0) > max_time:
                    max_time = seg["end"]
                
    if max_time == 0.0:
        return ""
        
    # 2. Divide timeline into bins of `time_bin_size`
    bins = []
    current_time = 0.0
    while current_time < max_time:
        bins.append((current_time, current_time + time_bin_size))
        current_time += time_bin_size
        
    srt_lines = []
    seq_idx = 1
    
    def format_ts(seconds):
        import datetime
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int(td.microseconds / 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
        
    for start, end in bins:
        bin_texts = {}
        has_content = False
        
        for model_name, segments in model_results.items():
            # Find all segments that overlap with this bin
            overlapping_texts = []
            if segments and isinstance(segments, list):
                for seg in segments:
                    if not isinstance(seg, dict): continue
                    # check if overlap: max(start, seg_start) < min(end, seg_end)
                    seg_start = seg.get("start", 0.0)
                    seg_end = seg.get("end", 0.0)
                    if max(start, seg_start) < min(end, seg_end):
                        overlapping_texts.append(seg.get("text", "").strip())
            
            joined_text = " ".join([t for t in overlapping_texts if t])
            if joined_text:
                bin_texts[model_name] = joined_text
                has_content = True
            else:
                bin_texts[model_name] = "-"
                
        if has_content:
            srt_lines.append(f"{seq_idx}")
            srt_lines.append(f"{format_ts(start)} --> {format_ts(end)}")
            # Display each model's output
            for model_name, text in bin_texts.items():
                short_name = model_name.split("/")[-1].replace(".nemo", "")
                srt_lines.append(f"[{short_name}]: {text}")
            srt_lines.append("") # empty line after segment
            seq_idx += 1
            
    return "\n".join(srt_lines)

def generate_comparison_txt(model_results):
    """
    Combines full text from multiple models into a single comparative text file.
    model_results: dict mapping model_name to dict with "text" key
    """
    lines = ["=== COMPARATIVE TRANSCRIPTIONS ===", ""]
    for model_name, res in model_results.items():
        short_name = model_name.split("/")[-1].replace(".nemo", "")
        text = res.get("text", "").strip() if isinstance(res, dict) else str(res).strip()
        lines.append(f"[{short_name.upper()}]:")
        lines.append(text)
        lines.append("-" * 40)
        lines.append("")
    return "\n".join(lines)

def split_audio_robust(file_path, chunk_length_s=600, overlap_s=10):
    """
    Robustly splits audio using FFmpeg with PCM encoding.
    Supports overlapping segments to prevent transcription gaps.
    """
    import hashlib
    import subprocess
    import glob
    
    logger.info(f"Splitting audio: {file_path} (len={chunk_length_s}s, overlap={overlap_s}s)")
    
    # 1. Prepare hash-based temp dir
    file_hash = hashlib.md5(str(file_path).encode('utf-8')).hexdigest()[:8]
    temp_dir = os.path.join(DOWNLOADS_DIR, f"temp_split_{file_hash}")
    
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 2. Get duration
    try:
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(file_path)]
        duration = float(subprocess.check_output(probe_cmd).decode().strip())
    except Exception as e:
        logger.warning(f"Failed to probe duration: {e}. Defaulting to full file.")
        duration = 0 # Full file treatment below

    # 3. Always normalize to 16kHz Mono if it's the only chunk or if it's the original
    if duration < chunk_length_s:
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
            return [output_path], temp_dir
        except Exception as e:
            logger.error(f"Normalization failed for {file_path}: {e}")
            return [str(file_path)], temp_dir # Fallback to original

    # 3. Manual overlap logic (FFmpeg segment doesn't support easy overlap)
    chunks = []
    start = 0
    idx = 0
    while start < duration:
        output_path = os.path.join(temp_dir, f"chunk_{idx:03d}.wav")
        # We take a piece of length chunk_length_s + overlap_s
        # BUT we only advance the 'start' by chunk_length_s
        actual_length = chunk_length_s + overlap_s
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-t', str(actual_length),
            '-i', str(file_path),
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
            logger.error(f"Failed to create chunk {idx}: {e}")
            
        start += chunk_length_s
        idx += 1
        if start >= duration:
            break

    if not chunks:
        return [str(file_path)], temp_dir
    return chunks, temp_dir

class ChunkItem:
    def __init__(self, file_path, start_s, end_s):
        self.file_path = file_path
        self.start_s = start_s
        self.end_s = end_s

def split_audio_vad(file_path, min_silence_len=1000, silence_thresh_offset=16, max_segment_len_s=30):
    """
    Splits audio using pydub's detect_nonsilent.
    Returns: list of ChunkItem objects, temp_dir path
    """
    import hashlib
    import shutil
    import subprocess
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    
    logger.info(f"VAD splitting audio: {file_path}")
    
    # 1. Prepare hash-based temp dir
    file_hash = hashlib.md5(str(file_path).encode('utf-8')).hexdigest()[:8]
    temp_dir = os.path.join(DOWNLOADS_DIR, f"temp_split_{file_hash}")
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 2. Convert to wav first if needed (pydub operates on wav/mp3/etc.)
    # We always convert/normalize to 16kHz mono wav for ASR consistency
    normalized_wav = os.path.join(temp_dir, "normalized_input.wav")
    cmd = [
        'ffmpeg', '-y',
        '-i', str(file_path),
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        normalized_wav
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    
    # 3. Load audio segment
    audio = AudioSegment.from_wav(normalized_wav)
    duration_s = len(audio) / 1000.0
    
    # 4. Detect speech ranges
    # We use a silence threshold relative to the audio's average dBFS
    silence_thresh = max(-45, audio.dBFS - silence_thresh_offset)
    ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    # If no speech found, fallback to treating the entire file as one segment
    if not ranges:
        logger.warning("VAD found no speech segments. Falling back to full file.")
        ranges = [(0, len(audio))]
        
    # Post-process ranges to split very long segments (VAD might not find pauses for a long time)
    processed_ranges = []
    max_segment_ms = max_segment_len_s * 1000
    for start, end in ranges:
        current_start = start
        while current_start < end:
            current_end = min(end, current_start + max_segment_ms)
            processed_ranges.append((current_start, current_end))
            current_start = current_end
            
    # 5. Export chunks
    chunks = []
    for idx, (start_ms, end_ms) in enumerate(processed_ranges):
        output_path = os.path.join(temp_dir, f"chunk_{idx:03d}.wav")
        
        # Add a tiny padding on both sides to prevent cutting off words
        pad_ms = 150
        actual_start_ms = max(0, start_ms - pad_ms)
        actual_end_ms = min(len(audio), end_ms + pad_ms)
        
        chunk_segment = audio[actual_start_ms:actual_end_ms]
        chunk_segment.export(output_path, format="wav")
        
        # Attach the actual global speech timestamps (excluding padding)
        chunks.append(ChunkItem(
            file_path=output_path,
            start_s=start_ms / 1000.0,
            end_s=end_ms / 1000.0
        ))
        
    logger.info(f"VAD splitting complete: generated {len(chunks)} chunks.")
    return chunks, temp_dir

