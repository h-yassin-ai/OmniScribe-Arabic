import sys
import os
import shutil
import logging

# Ensure we are in the root
if os.path.exists("modules"):
    sys.path.append(os.getcwd())
else:
    print("Please run from project root")
    sys.exit(1)

# Configure logging to stdout
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from modules.media_handler import get_video_info, download_audio

url = "https://www.facebook.com/reel/1345884746890365"
output_dir = "debug_output"

if os.path.exists(output_dir):
    try:
        shutil.rmtree(output_dir)
    except:
        pass
os.makedirs(output_dir, exist_ok=True)

print(f"--- 1. Getting Metadata for: {url} ---")
try:
    infos = get_video_info(url)
    print(f"Metadata Result: {infos}")
except Exception as e:
    print(f"Metadata Failed: {e}")
    sys.exit(1)

if not infos:
    print("No info returned!")
    sys.exit(1)

video_info = infos[0]
print(f"--- 2. Downloading Video (ID: {video_info['id']}) ---")

try:
    path = download_audio(video_info['url'], video_id=video_info['id'], output_dir=output_dir)
    print(f"Download Result Path: {path}")
    
    if path and os.path.exists(path):
        print(f"SUCCESS: File exists at {path}")
        print(f"Size: {os.path.getsize(path)} bytes")
    else:
        print("FAILURE: Path returned but file missing or None")
        
except Exception as e:
    print(f"Download Failed: {e}")
    import traceback
    traceback.print_exc()
