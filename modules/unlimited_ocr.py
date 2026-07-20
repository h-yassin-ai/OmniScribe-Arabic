import os
import sys
import tempfile
import time
import shutil
import fitz  # PyMuPDF
import math
import re
import gradio as gr
from modules.utils import logger, OUTPUTS_DIR

_model = None
_tokenizer = None
ocr_token_queue = None

# Base directory for OCR runs
OCR_RUNS_DIR = os.path.join(OUTPUTS_DIR, "ocr_runs")
os.makedirs(OCR_RUNS_DIR, exist_ok=True)

def get_pdf_workspace_dir(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(OCR_RUNS_DIR, pdf_name)

def get_ocr_text_path(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(get_pdf_workspace_dir(pdf_path), f"{pdf_name}.txt")

def clean_ocr_text(text):
    # Strip any <|det|>...<|/det|> coordinate blocks
    pattern = r'<\|det\|>.*?<\|/det\|>'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL)
    
    # Strip other VLM-specific tags if any remain
    cleaned = cleaned.replace('<|ref|>', '').replace('<|/ref|>', '')
    cleaned = cleaned.replace('<PAGE>', '')
    
    # Standard cleanup of multiple newlines and spacing
    lines = []
    for line in cleaned.split('\n'):
        line_stripped = line.strip()
        if line_stripped:
            lines.append(line_stripped)
    return '\n'.join(lines)

def get_initial_ocr_structure(total_pages):
    combined = []
    for i in range(1, total_pages + 1):
        combined.append(f"--- PAGE {i} ---\n[Non transcrit - Lancer l'OCR pour transcrire]")
    return "\n\n".join(combined)

def patch_tps_streamer():
    import sys
    modeling_unlimitedocr_module = None
    for key in sys.modules:
        if "modeling_unlimitedocr" in key:
            modeling_unlimitedocr_module = sys.modules[key]
            break
            
    if modeling_unlimitedocr_module is not None:
        if not hasattr(modeling_unlimitedocr_module, '_patched_tps'):
            original_tps = modeling_unlimitedocr_module.TPSTextStreamer
            
            class CustomGradioStreamer(original_tps):
                def on_finalized_text(self, text: str, stream_end: bool = False):
                    global ocr_token_queue
                    if ocr_token_queue is not None:
                        ocr_token_queue.put(text)
                    super().on_finalized_text(text, stream_end)
                    
            modeling_unlimitedocr_module.TPSTextStreamer = CustomGradioStreamer
            modeling_unlimitedocr_module._patched_tps = True
            logger.info("[OCR Patch] TPSTextStreamer monkey-patched successfully for Gradio live streaming.")

def _init_model(progress=None):
    global _model, _tokenizer
    if _model is not None:
        return
        
    if progress:
        progress(0.1, desc="Initialisation et application des patchs...")
        
    import httpx
    import requests
    import urllib3
    
    # Configure Hugging Face home to cache in workspace
    os.environ['HF_HOME'] = r'J:\UnivHm_NextJS\Al-Kutub-Automator\.cache\huggingface'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    # Disable SSL verification in huggingface_hub using requests backend
    try:
        from huggingface_hub import configure_http_backend
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        def backend_factory() -> requests.Session:
            session = requests.Session()
            session.verify = False
            return session
        configure_http_backend(backend_factory=backend_factory)
    except Exception as e:
        logger.warning(f"[OCR Init] huggingface_hub SSL bypass setup failed: {e}")
        
    try:
        original_send = requests.Session.send
        def new_send(self, request, **kwargs):
            kwargs['verify'] = False
            return original_send(self, request, **kwargs)
        requests.Session.send = new_send
    except Exception as e:
        logger.warning(f"[OCR Init] requests Session.send bypass setup failed: {e}")
        
    # Dynamically patch transformers for version compatibility
    import transformers
    import transformers.utils
    import transformers.utils.import_utils
    
    if not hasattr(transformers.utils.import_utils, 'is_torch_fx_available'):
        transformers.utils.import_utils.is_torch_fx_available = lambda: True
    if not hasattr(transformers.utils, 'is_torch_fx_available'):
        transformers.utils.is_torch_fx_available = lambda: True
        
    def pretrained_config_getattr(self, name):
        defaults = {
            'attention_dropout': 0.0,
            'attention_bias': False,
            'rms_norm_eps': 1e-6,
            'rope_theta': 10000.0,
            'rope_scaling': None,
            'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000.0},
            'pretraining_tp': 1,
            'pad_token_id': None,
            'bos_token_id': 0,
            'eos_token_id': 1,
            'hidden_act': 'silu',
            'hidden_dropout': 0.0,
            'initializer_range': 0.02,
            'use_cache': True,
            'tie_word_embeddings': False,
            'moe_layer_freq': 1,
            'aux_loss_alpha': 0.001,
            'norm_topk_prob': True,
            'routed_scaling_factor': 1.0,
            'scoring_func': "softmax",
            'seq_aux': True,
            'ep_size': 1,
            'cache_implementation': None,
            '_attn_implementation': 'eager',
            '_pre_quantization_dtype': None,
            '_ring_window': None,
        }
        if name in defaults:
            return defaults[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    transformers.PretrainedConfig.__getattr__ = pretrained_config_getattr
    
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    
    if progress:
        progress(0.3, desc="Téléchargement / Chargement du modèle Unlimited-OCR sur le GPU...")
        
    try:
        config = AutoConfig.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True)
        _tokenizer = AutoTokenizer.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True)
        _model = AutoModel.from_pretrained(
            "baidu/Unlimited-OCR",
            config=config,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        ).eval().cuda()
        logger.info("[OCR Init] Model loaded successfully on GPU.")
        
        # Apply the streaming patch
        patch_tps_streamer()
    except Exception as e:
        logger.error(f"[OCR Init] Failed to load model: {e}")
        raise e

def get_cached_pdfs():
    import glob
    path = os.path.join(OCR_RUNS_DIR, "*", "*.pdf")
    files = glob.glob(path)
    return [os.path.basename(f) for f in sorted(files)]

def get_pdf_total_pages(pdf_path):
    if not pdf_path or not os.path.exists(pdf_path):
        return 0
    doc = fitz.open(pdf_path)
    total = len(doc)
    doc.close()
    return total

def prepare_pdf_cache(pdf_path, progress=None):
    if not pdf_path or not os.path.exists(pdf_path):
        return None, 0
        
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    workspace_dir = get_pdf_workspace_dir(pdf_path)
    os.makedirs(workspace_dir, exist_ok=True)
    
    # Copy PDF to workspace if it is not already there
    dest_path = os.path.join(workspace_dir, f"{pdf_name}.pdf")
    if os.path.abspath(pdf_path) != os.path.abspath(dest_path):
        shutil.copy2(pdf_path, dest_path)
        pdf_path = dest_path
        
    images_dir = os.path.join(workspace_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Check if page images already exist
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    images_exist = True
    for i in range(1, total_pages + 1):
        page_img = os.path.join(images_dir, f"page_{i:04d}.png")
        if not os.path.exists(page_img):
            images_exist = False
            break
            
    if not images_exist:
        if progress:
            progress(0.1, desc="Génération du cache des pages du PDF...")
        dpi = 150
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for idx in range(total_pages):
            page_img = os.path.join(images_dir, f"page_{idx + 1:04d}.png")
            doc[idx].get_pixmap(matrix=mat).save(page_img)
            if progress:
                progress(0.1 + 0.8 * ((idx + 1) / total_pages), desc=f"Rendu de la page {idx+1}/{total_pages}...")
                
    doc.close()
    return pdf_path, total_pages

def load_parsed_ocr(pdf_path):
    ocr_path = get_ocr_text_path(pdf_path)
    if not os.path.exists(ocr_path):
        return {}
        
    pages = {}
    try:
        with open(ocr_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        parts = content.split("--- PAGE ")
        for part in parts[1:]:
            if not part.strip():
                continue
            lines = part.split("\n", 1)
            header = lines[0].strip()
            page_num_str = header.replace("---", "").strip()
            try:
                page_num = int(page_num_str)
                page_text = lines[1] if len(lines) > 1 else ""
                pages[page_num] = page_text.strip("\r\n")
            except ValueError:
                pass
    except Exception as e:
        logger.error(f"Error parsing OCR file: {e}")
        
    return pages

def save_parsed_ocr(pdf_path, pages_dict):
    ocr_path = get_ocr_text_path(pdf_path)
    os.makedirs(os.path.dirname(ocr_path), exist_ok=True)
    
    combined = []
    for page_num in sorted(pages_dict.keys()):
        text = pages_dict[page_num]
        combined.append(f"--- PAGE {page_num} ---\n{text}")
        
    with open(ocr_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(combined))
        
    return ocr_path

def run_ocr_batch(pdf_path, start_page, end_page, prompt="Multi page parsing.", mode="Base (Précis, pleine résolution)"):
    if not pdf_path or not os.path.exists(pdf_path):
        return "Erreur : PDF introuvable."
        
    start_page = int(start_page)
    end_page = int(end_page)
    
    batch_size = end_page - start_page + 1
    if batch_size <= 0:
        return "Erreur : La page de fin doit être supérieure ou égale à la page de début."
    if batch_size > 15:
        return "Erreur : La taille du lot ne peut pas dépasser 15 pages pour éviter les hallucinations."
        
    try:
        # Ensure model is initialized
        _init_model()
        
        workspace_dir = get_pdf_workspace_dir(pdf_path)
        images_dir = os.path.join(workspace_dir, "images")
        
        # Gather the image file paths for this batch
        image_files = []
        for page_num in range(start_page, end_page + 1):
            page_img = os.path.join(images_dir, f"page_{page_num:04d}.png")
            if not os.path.exists(page_img):
                return f"Erreur : Image de la page {page_num} introuvable dans le cache."
            image_files.append(page_img)
            
        # Load existing OCR results
        pages_dict = load_parsed_ocr(pdf_path)
        
        temp_out_dir = tempfile.mkdtemp(prefix=f"ocr_multi_batch_{start_page}_{end_page}_")
        try:
            # Multi-page PDF parsing uses base (image_size=1024)
            # Prompt must start with <image>
            full_prompt = prompt if prompt.startswith("<image>") else f"<image>{prompt}"
            
            outputs, _ = _model.infer_multi(
                _tokenizer,
                prompt=full_prompt,
                image_files=image_files,
                output_path=temp_out_dir,
                image_size=1024,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=1024,
                save_results=True
            )
            
            # Parse outputs: separated by <PAGE>
            pages = outputs.split('<PAGE>')[1:]
            for idx, page_output in enumerate(pages):
                page_num = start_page + idx
                if page_num <= end_page:
                    # Clean coordinates out of the text
                    pages_dict[page_num] = clean_ocr_text(page_output)
                    
            # Save the updated dict
            save_parsed_ocr(pdf_path, pages_dict)
            
        finally:
            shutil.rmtree(temp_out_dir, ignore_errors=True)
            
        # Free CUDA cache
        import gc
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        
        return f"✅ OCR complété pour les pages {start_page} à {end_page}."
        
    except Exception as e:
        logger.error(f"Error in run_ocr_batch: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Échec de l'OCR : {str(e)}"
