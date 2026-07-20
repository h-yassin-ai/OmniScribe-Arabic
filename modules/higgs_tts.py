import os
import gc
import torch
import torchaudio
import logging
import datetime

from modules.ocr_common import configure_huggingface_http, configure_transformers_compat
from modules.utils import get_output_path

logger = logging.getLogger(__name__)

def generate_voice_higgs(text, ref_audio_path=None, ref_text=None, temperature=0.7, top_p=0.95):
    """
    Loads Higgs TTS 3 4B, generates speech (optionally with zero-shot voice cloning),
    saves it to a temporary WAV file, unloads the model, and returns the path.
    """
    if not torch.cuda.is_available():
        return None, "Erreur: CUDA (GPU) n'est pas disponible pour ce modèle."

    # OCR models remain cached between page runs; release them before loading
    # the 5B TTS model to avoid an avoidable out-of-memory error.
    try:
        from modules.qari_ocr import free_model as free_qari
        free_qari()
    except Exception:
        pass

    configure_huggingface_http()
    configure_transformers_compat()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    repo = "multimodalart/higgs-audio-v3-tts-4b-transformers"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(get_output_path(f"higgs_{timestamp}", "wav", subfolder="tts"))
    
    logger.info("Loading Higgs TTS model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to("cuda").eval()
        
        generate_kwargs = {
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Load reference audio if provided
        if ref_audio_path and os.path.exists(ref_audio_path):
            logger.info(f"Loading reference audio for voice cloning: {ref_audio_path}")
            ref_wav, sr = torchaudio.load(ref_audio_path)
            
            # Higgs generate_speech accepts reference_audio and reference_sample_rate
            # Ensure ref_wav is 1D or 2D [1, samples]
            if ref_wav.ndim > 1:
                # Average channels if stereo
                ref_wav = ref_wav.mean(dim=0)
            
            generate_kwargs["reference_audio"] = ref_wav.to("cuda", dtype=torch.bfloat16)
            generate_kwargs["reference_sample_rate"] = sr
            if ref_text and ref_text.strip():
                generate_kwargs["reference_text"] = ref_text.strip()
                
        logger.info(f"Generating speech for text: {text}")
        with torch.no_grad():
            wav = model.generate_speech(
                text,
                tokenizer,
                **generate_kwargs
            )
            
        # Save output wav
        waveform = wav.detach().to(dtype=torch.float32, device="cpu")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(output_path, waveform, model.config.sample_rate)
        logger.info(f"Speech saved successfully to {output_path}")
        return output_path, "Génération réussie !"
        
    except Exception as e:
        logger.error(f"Error during Higgs TTS generation: {e}")
        return None, f"Erreur : {str(e)}"
    finally:
        logger.info("Unloading Higgs model to free up VRAM...")
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
