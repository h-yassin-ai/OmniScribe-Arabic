import os
import gc
import torch
import torchaudio
import logging
import datetime

from modules.ocr_common import configure_huggingface_http, configure_transformers_compat
from modules.utils import get_output_path

logger = logging.getLogger(__name__)

import shutil
import tempfile

def load_reference_audio(audio_path, target_sr=24000):
    """
    Loads reference audio safely on Windows, converts to mono [1, samples],
    resamples to 24kHz, and normalizes audio volume to prevent empty/noisy output.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if os.path.isdir(audio_path):
        raise IsADirectoryError(f"Audio path is a directory: {audio_path}")

    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, f"higgs_ref_{os.getpid()}_{os.path.basename(audio_path)}")
    try:
        shutil.copy2(audio_path, temp_file)
        wav, sr = torchaudio.load(temp_file)

        # Ensure 2D tensor format [channels, samples]
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        # Convert stereo or multi-channel audio to mono [1, samples]
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample to 24kHz if sample rate differs
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wav = resampler(wav)
            sr = target_sr

        # Normalize volume to 90% peak to prevent quiet microphone recordings
        max_val = torch.max(torch.abs(wav))
        if max_val > 1e-6:
            wav = (wav / max_val) * 0.9

        return wav, sr
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

def generate_voice_higgs(text, ref_audio_path=None, ref_text=None, temperature=0.7, top_p=0.95, status_callback=None):
    """
    Loads Higgs TTS 3 4B, generates speech (optionally with zero-shot voice cloning),
    saves it to a temporary WAV file, unloads the model, and returns the path.
    """
    def log_status(msg):
        logger.info(msg)
        if status_callback:
            try:
                status_callback(msg)
            except Exception:
                pass

    if not torch.cuda.is_available():
        return None, "Erreur: CUDA (GPU) n'est pas disponible pour ce modèle."

    log_status("🧹 Libération des autres modèles pour optimiser la VRAM GPU...")
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
    
    log_status("⏳ Chargement du modèle Higgs TTS 3 (4B) dans la mémoire VRAM GPU (cela peut prendre 10 à 25s)...")
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
            log_status(f"🎙️ Chargement et traitement de votre voix de référence...")
            ref_wav, sr = load_reference_audio(ref_audio_path)
            
            generate_kwargs["reference_audio"] = ref_wav
            generate_kwargs["reference_sample_rate"] = sr
            if ref_text and ref_text.strip():
                generate_kwargs["reference_text"] = ref_text.strip()
                
        log_status(f"🗣️ Génération du signal vocal en arabe en cours...")
        with torch.no_grad():
            wav = model.generate_speech(
                text,
                tokenizer,
                **generate_kwargs
            )
            
        log_status("💾 Sauvegarde et création du fichier audio WAV final...")
        waveform = wav.detach().to(dtype=torch.float32, device="cpu")
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        sample_rate = getattr(model.config, "sample_rate", 24000)
        torchaudio.save(output_path, waveform, sample_rate)
        log_status(f"✅ Synthèse vocale terminée avec succès ! Fichier : {output_path}")
        return output_path, "Génération réussie !"
        
    except Exception as e:
        error_msg = f"Erreur : {str(e)}"
        log_status(f"❌ {error_msg}")
        return None, error_msg
    finally:
        log_status("🧹 Nettoyage de la mémoire VRAM GPU...")
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
