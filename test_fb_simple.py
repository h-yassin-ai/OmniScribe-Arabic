import sys
import os
import shutil

# Ensure we are in the root
if os.path.exists("modules"):
    sys.path.append(os.getcwd())
else:
    print("Please run from project root")
    sys.exit(1)

from modules.media_handler import download_facebook

url = "https://www.facebook.com/reel/1345884746890365"
output_dir = "tests_output_simple"

if os.path.exists(output_dir):
    try: shutil.rmtree(output_dir)
    except: pass
os.makedirs(output_dir, exist_ok=True)

print(f"Testing download for: {url}")
try:
    path = download_facebook(url, output_dir=output_dir)
    if path and os.path.exists(path):
        print(f"Success! File downloaded to: {path}")
    else:
        print("Function returned but file missing?")
except Exception as e:
    print(f"Download execution failed: {e}")
