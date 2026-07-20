import yt_dlp
import os
import glob
import subprocess
import re
import time
import random
import json
from .utils import logger, DOWNLOADS_DIR, BASE_DIR

def get_po_token_data():
    """
    Reads PO token and visitor data from local files.
    Returns: (token, visitor_data)
    """
    po_token_file = BASE_DIR / "po_token.txt"
    visitor_data_file = BASE_DIR / "visitor_data.txt"
    
    po_token = None
    visitor_data = None
    
    if po_token_file.exists():
        with open(po_token_file, "r") as f:
            po_token = f.read().strip()
    
    if visitor_data_file.exists():
        with open(visitor_data_file, "r") as f:
            visitor_data = f.read().strip()
            
    return po_token, visitor_data

def get_cookies_option():
    cookie_file = BASE_DIR / "cookies.txt"
    if cookie_file.exists():
        return {'cookiefile': str(cookie_file)}
    else:
        logger.info("No cookies.txt found, trying to read from Chrome...")
        return {'cookies_from_browser': ('chrome',)}

def get_video_info(url, proxy=None):
    if is_facebook_url(url):
        # Try to fetch metadata using yt-dlp first
        try:
            ydl_opts_fb = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'dump_single_json': True,
            }
            if proxy: ydl_opts_fb['proxy'] = proxy
            
            with yt_dlp.YoutubeDL(ydl_opts_fb) as ydl:
                info = ydl.extract_info(url, download=False)
                if info:
                    return [{
                        'id': info.get('id', 'fb_video'),
                        'title': info.get('title', 'Facebook Video'),
                        'url': url,
                        'duration': info.get('duration')
                    }]
        except Exception:
            pass # Fallback to dummy if yt-dlp extraction fails but URL is valid
            
        return [{
             'id': 'fb_' + str(int(time.time())),
             'title': 'Facebook Video (Metadata Pending)',
             'url': url,
             'duration': None
         }]

    po_token, visitor_data = get_po_token_data()
    
    # Use 'web.gvs+' prefix as it's the most common for metadata extraction
    full_token = None
    if po_token:
        # If the user already provided a prefix, don't double it
        if "+" in po_token:
            full_token = po_token
        else:
            full_token = f"web.gvs+{po_token}"

    ydl_opts = {
        'extract_flat': True,
        'dump_single_json': True,
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'nocheckcertificate': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'extra_headers': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://www.google.com/',
        },
        'extractor_args': {
            'youtube': {
                'player_client': ['web', 'android'],
                'po_token': [full_token] if full_token else None,
            }
        },
    }
    
    cookie_opts = get_cookies_option()
    ydl_opts.update(cookie_opts)
    
    if proxy:
        ydl_opts['proxy'] = proxy

    logger.info(f"Fetching metadata for: {url}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=False)
            
            if not result:
                match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
                if match:
                    v_id = match.group(1)
                    return [{
                        'id': v_id,
                        'title': "Video (ID Extracted from URL)",
                        'url': f"https://www.youtube.com/watch?v={v_id}",
                        'duration': None
                    }]
                return []
                
            res_type = result.get('_type', 'video')
            entries = result.get('entries', [])
            
            if (res_type == 'playlist' or entries) and len(entries) > 0:
                videos = []
                for entry in entries:
                    if not entry: continue
                    v_id = entry.get('id')
                    if not v_id or len(v_id) != 11:
                        continue
                    videos.append({
                        'id': v_id,
                        'title': entry.get('title') or "Untitled Video",
                        'url': f"https://www.youtube.com/watch?v={v_id}",
                        'duration': entry.get('duration')
                    })
                return videos
            else:
                v_id = result.get('id')
                if not v_id or len(v_id) != 11:
                    match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
                    if match:
                        v_id = match.group(1)
                    else:
                        return []
                    
                return [{
                    'id': v_id,
                    'title': result.get('title') if len(result.get('id', '')) == 11 else "Video (Partial Playlist Link)",
                    'url': f"https://www.youtube.com/watch?v={v_id}",
                    'duration': result.get('duration')
                }]
                
    except Exception as e:
        logger.error(f"Error fetching metadata: {str(e)}")
        raise e

def convert_to_wav(input_path, output_path):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True

    cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-ac', '1',
        '-ar', '16000',
        '-loglevel', 'error',
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def find_existing_file(video_id, output_dir):
    if not video_id or len(video_id) != 11:
        return None
    try:
        files = os.listdir(output_dir)
        for f in files:
            if video_id in f:
                full_path = os.path.join(output_dir, f)
                if os.path.isdir(full_path):
                    continue
                return full_path
    except OSError:
        pass
    return None

def download_audio(url, video_id=None, output_dir=DOWNLOADS_DIR, proxy=None):
    # Check for Facebook URL first to avoid YouTube ID logic messing it up
    if is_facebook_url(url):
        # Determine specific Facebook ID if possible (for logging/filename)
        # But pass original URL to downloader
        return download_facebook(url, output_dir, proxy)

    if not video_id:
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
        if match:
            video_id = match.group(1)
            
    if video_id:
        url = f"https://www.youtube.com/watch?v={video_id}"

    if video_id:
        existing = find_existing_file(video_id, output_dir)
        if existing:
            if existing.lower().endswith(".wav"):
                return existing
            wav_path = os.path.splitext(existing)[0] + ".wav"
            if convert_to_wav(existing, wav_path):
                 return wav_path
            else:
                 return None

    po_token, visitor_data = get_po_token_data()

    # Strategy: Use iOS client as primary for downloads, it's the most stable with PO Token
    # Needs prefix 'ios.gvs+' or 'android.gvs+'
    full_token = None
    if po_token:
        # If user didn't prefix, we assume ios.gvs+ as our primary strategy
        if "+" not in po_token:
            full_token = f"ios.gvs+{po_token}"
        else:
            full_token = po_token

    params = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(title)s [%(id)s].%(ext)s'),
        'quiet': False,
        'overwrites': False,
        'noplaylist': True,
        'nocheckcertificate': True,
        'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
        'extra_headers': {
            'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://www.google.com/',
        },
        'extractor_args': {
            'youtube': {
                'player_client': ['ios', 'android'],
                'po_token': [full_token] if full_token else None,
            }
        },
    }
    
    cookie_opts = get_cookies_option()
    params.update(cookie_opts)

    if proxy:
        params['proxy'] = proxy
    
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading (Attempt {attempt+1}/{max_retries}): {url}")
            
            # Anti-ban delay
            time.sleep(random.uniform(5, 10))

            with yt_dlp.YoutubeDL(params) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)
                
                if not os.path.exists(filename):
                    base = os.path.splitext(filename)[0]
                    for ext in ['.m4a', '.webm', '.opus', '.mp3', '.mp4']:
                        if os.path.exists(base + ext):
                            filename = base + ext
                            break
                
                if not os.path.exists(filename):
                    continue
                    
                wav_filename = os.path.splitext(filename)[0] + ".wav"
                if convert_to_wav(filename, wav_filename):
                    if filename != wav_filename and not filename.endswith(".mp4"):
                        try: os.remove(filename)
                        except OSError: pass
                    return wav_filename
                else:
                    return None
                    
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                # If we get a 403, try to switch client to android just in case
                if "403" in str(e):
                    params['extractor_args']['youtube']['player_client'] = ['android', 'web']
                    if full_token and "ios.gvs+" in full_token:
                         params['extractor_args']['youtube']['po_token'] = [full_token.replace("ios.gvs+", "android.gvs+")]
                
                time.sleep(15) # Heavier backoff
                continue
            break

    logger.error(f"Download Final Failure: {last_error}")
    raise last_error if last_error else Exception("Unknown download error")

def scan_local_folder(folder_path):
    """
    Scans a local folder for .wav, .mp3, .mp4 files.
    Returns: list of VideoItem-like dicts
    """
    if not os.path.isdir(folder_path):
        logger.error(f"Not a valid directory: {folder_path}")
        return []
        
    supported_exts = ('.wav', '.mp3', '.mp4', '.m4a', '.webm', '.opus')
    items = []
    
    # Sort files naturally
    files = sorted(os.listdir(folder_path))
    
    for filename in files:
        if filename.lower().endswith(supported_exts):
            full_path = os.path.join(folder_path, filename)
            # Create a "pseudo-vid" for the pipeline
            # We use the filename without ext as the "id" if possible, 
            # but usually we'll just use the full path as indicator
            file_id = re.sub(r'[^a-zA-Z0-9]', '_', os.path.splitext(filename)[0])[:11]
            
            items.append({
                'id': file_id,
                'title': os.path.splitext(filename)[0],
                'url': full_path, # Local path instead of URL
                'duration': None,
                'is_local': True
            })
            
    logger.info(f"Scanned local folder: found {len(items)} files.")
    return items

    logger.info(f"Scanned local folder: found {len(items)} files.")
    return items

def is_facebook_url(url):
    return "facebook.com" in url or "fb.watch" in url

def download_generic(url, output_dir=DOWNLOADS_DIR, proxy=None):
    """
    Generic download using yt-dlp with minimal configuration.
    """
    logger.info(f"Fallback: Initializing generic download for {url}")
    
    params = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(title)s [%(id)s].%(ext)s'),
        'quiet': False,
        'overwrites': False,
        'noplaylist': True,
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'restrictfilenames': True, # Ensure windows compatibility
        'trim_file_name': 50, # Limit filename length
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    }
    
    
    if proxy:
        params['proxy'] = proxy
        
    try:
        with yt_dlp.YoutubeDL(params) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                raise Exception("No info extracted")
                
            filename = ydl.prepare_filename(info)
            if not os.path.exists(filename):
                # Try to find the file with other extensions
                base = os.path.splitext(filename)[0]
                for ext in ['.m4a', '.webm', '.opus', '.mp3', '.mp4']:
                    if os.path.exists(base + ext):
                        filename = base + ext
                        break
            
            if not os.path.exists(filename):
                raise Exception("Downloaded file not found")
                
            # Convert to WAV
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            if convert_to_wav(filename, wav_filename):
                if filename != wav_filename and not filename.endswith(".mp4"):
                    try: os.remove(filename)
                    except OSError: pass
                from modules.utils import sanitize_path_prevent_dotfile
                return sanitize_path_prevent_dotfile(wav_filename)
            else:
                return None
    except Exception as e:
        logger.error(f"Generic download failed: {e}")
        raise e

def download_facebook(url, output_dir=DOWNLOADS_DIR, proxy=None):
    script_path = os.path.join(BASE_DIR, "modules", "fb_downloader.js")
    
    logger.info(f"Attempting Facebook download (Node.js): {url}")
    
    # Method 1: Try Node.js script (as requested)
    try:
        process = subprocess.Popen(
            ["node", script_path, url, str(output_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            # Try to parse JSON output
            try:
                lines = stdout.strip().split('\n')
                json_str = lines[-1]
                result = json.loads(json_str) 
                
                if "error" not in result:
                    video_path = result.get("path")
                    if video_path and os.path.exists(video_path):
                         # Convert to WAV
                        wav_path = os.path.splitext(video_path)[0] + ".wav"
                        if convert_to_wav(video_path, wav_path):
                            from modules.utils import sanitize_path_prevent_dotfile
                            return sanitize_path_prevent_dotfile(wav_path)
            except Exception:
                pass # Fallback to generic if parsing fails or invalid
                
        logger.warning(f"Node.js downloader failed or returned invalid data. Output: {stdout} Error: {stderr}")

    except Exception as e:
        logger.warning(f"Node.js downloader exception: {e}")

    # Method 2: Fallback to Generic yt-dlp
    logger.info("Falling back to generic yt-dlp for Facebook video...")
    return download_generic(url, output_dir, proxy)

