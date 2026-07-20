import os
import sys
import argparse
import torch
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

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using NeMo FastConformer.")
    parser.add_argument(
        "--audio", 
        type=str, 
        required=True, 
        help="Path to the audio file (.wav format recommended)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="stt_ar_fastconformer_hybrid_large_pc_v1.0.nemo", 
        help="Path to the local .nemo model file or HuggingFace/NGC model identifier"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="Device to run inference on (cuda or cpu)"
    )
    
    args = parser.parse_args()

    # Verify audio file exists
    if not os.path.exists(args.audio):
        print(f"[ERROR] Audio file not found: {args.audio}")
        sys.exit(1)

    print(f"Using device: {args.device}")
    
    # Set PyTorch device
    device = torch.device(args.device)

    # 2. Load the model (local .nemo file or pretrained from hub)
    print(f"Loading NeMo model from: {args.model} ...")
    try:
        if os.path.exists(args.model):
            # Attempt hybrid model loading
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
            # Load from HuggingFace / NGC Hub
            try:
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=args.model,
                    map_location=device
                )
            except Exception:
                model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=args.model
                )
                if hasattr(model, "to"):
                    model.to(device)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    print(f"Model loaded successfully: {model.__class__.__name__}")
    
    # 3. Configure Windows fixes if running on Windows
    if os.name == "nt" and hasattr(model, "cfg"):
        print("Applying Windows compatibility patches to config...")
        force_windows_safe_dataloader_cfg(model.cfg)

    # Move model to device and set to evaluation mode
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    # 4. Perform transcription
    print(f"Transcribing '{args.audio}'...")
    try:
        # Get signature to pass the right argument name
        import inspect
        sig = inspect.signature(model.transcribe)
        arg_name = "paths2audio_files" if "paths2audio_files" in sig.parameters else "audio"
        
        # Prepare arguments
        transcribe_kwargs = {
            arg_name: [args.audio],
            "batch_size": 1,
            "num_workers": 0
        }
        if "return_hypotheses" in sig.parameters:
            transcribe_kwargs["return_hypotheses"] = True
            
        with torch.inference_mode():
            res = model.transcribe(**transcribe_kwargs)
            
        if res and isinstance(res, list) and len(res) > 0:
            hypothesis = res[0]
            # If return_hypotheses=True, the result is a Hypothesis object or string
            transcript = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)
            
            print("\n" + "="*40)
            print("TRANSCRIPTION RESULT:")
            print("="*40)
            print(transcript)
            print("="*40 + "\n")
        else:
            print("[WARNING] Transcription completed, but returned no result.")
            
    except Exception as e:
        print(f"[ERROR] An error occurred during transcription: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
