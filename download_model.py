#!/usr/bin/env python3
"""Download script for NVIDIA NeMo FastConformer model."""

import os
import sys
import urllib.request

MODEL_URL = "https://huggingface.co/nvidia/stt_ar_fastconformer_hybrid_large_pc/resolve/main/stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo"
TARGET_FILENAME = "stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo"

def download_model(target_dir="."):
    target_path = os.path.join(target_dir, TARGET_FILENAME)
    if os.path.exists(target_path):
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"Model already present at {target_path} ({size_mb:.1f} MB). Skipping download.")
        return target_path

    print(f"Downloading NVIDIA NeMo FastConformer model to {target_path} ...")
    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
        sys.stdout.write(f"\rProgress: {percent}% ({count * block_size / (1024*1024):.1f} MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(MODEL_URL, target_path, reporthook=_progress)
    print("\nDownload complete!")
    return target_path

if __name__ == "__main__":
    download_model()
