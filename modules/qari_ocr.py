"""Arabic book OCR with Qari-OCR 0.4.0 (Qwen3-VL 4B + PEFT adapter)."""

from __future__ import annotations

import glob
import os
import tempfile
import time
from pathlib import Path

import fitz

from modules.ocr_common import (
    OCR_PROMPT,
    clean_generated_text,
    configure_huggingface_http,
    configure_transformers_compat,
    disable_hidden_tokenizer_truncation,
    load_page_map,
    model_device,
    release_model,
    render_pdf_cache,
    save_page_map,
    validate_page_range,
)
from modules.utils import OUTPUTS_DIR, logger


MODEL_ID = "NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct"
BASE_MODEL = "unsloth/Qwen3-VL-4B-Instruct"
OCR_RUNS_DIR = os.path.join(OUTPUTS_DIR, "ocr_runs")
os.makedirs(OCR_RUNS_DIR, exist_ok=True)

_model = None
_processor = None
_fast_vision_model = None


def _inference_precision() -> str:
    """Return the requested runtime precision; the application defaults to 8-bit."""
    value = os.getenv("QARI_INFERENCE_PRECISION", "8bit").strip().lower()
    if value in {"bf16", "bfloat16", "16bit"}:
        return "bf16"
    return "8bit"


def _cached_snapshot(repo_id: str) -> str:
    """Use an existing project snapshot so Unsloth never redownloads Qari."""
    cache_root = Path(os.environ["HF_HOME"]) / "hub"
    repo_folder = cache_root / f"models--{repo_id.replace('/', '--')}"
    ref = repo_folder / "refs" / "main"
    if ref.exists():
        snapshot = repo_folder / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if snapshot.exists():
            return str(snapshot.resolve())
    return repo_id


def initialize_unsloth_runtime():
    """Import and initialize Unsloth before Transformers loads.

    Unsloth Zoo probes the Hugging Face cache with NamedTemporaryFile. On this
    Windows/J: workspace that probe can block for minutes because delete-on-close
    locking is unsupported by the volume. The targeted wrapper only disables
    delete-on-close for that cache probe and removes the file immediately after.
    """
    global _fast_vision_model
    if _fast_vision_model is not None:
        return _fast_vision_model

    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("TRITON_CACHE_DIR", str(project_root / ".cache" / "triton"))
    os.environ.setdefault(
        "UNSLOTH_COMPILE_LOCATION",
        str(project_root / ".cache" / "unsloth_compiled"),
    )
    Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["UNSLOTH_COMPILE_LOCATION"]).mkdir(parents=True, exist_ok=True)

    configure_transformers_compat()
    original_named_temporary_file = tempfile.NamedTemporaryFile
    hf_home = Path(os.environ["HF_HOME"]).resolve()

    class _DeletingTemporaryFile:
        def __init__(self, wrapped):
            self._wrapped = wrapped
            self.name = wrapped.name

        def __enter__(self):
            return self._wrapped.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            try:
                return self._wrapped.__exit__(exc_type, exc_value, traceback)
            finally:
                try:
                    os.unlink(self.name)
                except OSError:
                    pass

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

    def named_temporary_file_without_delete_lock(*args, **kwargs):
        directory = kwargs.get("dir")
        if directory is not None:
            try:
                if Path(directory).resolve().is_relative_to(hf_home):
                    kwargs["delete"] = False
                    kwargs.pop("delete_on_close", None)
                    return _DeletingTemporaryFile(
                        original_named_temporary_file(*args, **kwargs)
                    )
            except (OSError, ValueError):
                pass
        return original_named_temporary_file(*args, **kwargs)

    tempfile.NamedTemporaryFile = named_temporary_file_without_delete_lock
    try:
        from unsloth import FastVisionModel
    finally:
        tempfile.NamedTemporaryFile = original_named_temporary_file

    _fast_vision_model = FastVisionModel
    logger.info(
        "[Qari OCR] Unsloth runtime initialized (Triton cache: %s).",
        os.environ["TRITON_CACHE_DIR"],
    )
    return _fast_vision_model


def get_pdf_workspace_dir(pdf_path):
    return os.path.join(OCR_RUNS_DIR, os.path.splitext(os.path.basename(pdf_path))[0])


def get_cached_pdfs():
    files = glob.glob(os.path.join(OCR_RUNS_DIR, "*", "*.pdf"))
    return [os.path.basename(path) for path in sorted(files)]


def get_pdf_total_pages(pdf_path):
    if not pdf_path or not os.path.exists(pdf_path):
        return 0
    with fitz.open(pdf_path) as document:
        return len(document)


def prepare_pdf_cache(pdf_path, progress=None):
    if not pdf_path or not os.path.exists(pdf_path):
        return None, 0
    return render_pdf_cache(pdf_path, get_pdf_workspace_dir(pdf_path), progress)


def get_ocr_text_path(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(get_pdf_workspace_dir(pdf_path), f"{pdf_name}_qari.txt")


def clean_ocr_text(text):
    return clean_generated_text(text)


def free_model():
    global _model, _processor
    model = _model
    _model = None
    _processor = None
    release_model(model)
    logger.info("[Qari OCR] Model released.")


def _load_processor(AutoProcessor):
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
    except Exception as exc:
        logger.warning("[Qari OCR] Adapter processor unavailable; using base processor: %s", exc)
        processor = AutoProcessor.from_pretrained(BASE_MODEL)
    return disable_hidden_tokenizer_truncation(processor)


def _load_model(FastVisionModel, torch):
    if not torch.cuda.is_available():
        raise RuntimeError("Qari avec Unsloth nécessite un GPU CUDA.")

    adapter_source = _cached_snapshot(MODEL_ID)
    precision = _inference_precision()
    load_kwargs = {
        "tokenizer_name": adapter_source,
        "max_seq_length": int(os.getenv("QARI_MAX_SEQ_LENGTH", "16384")),
        "load_in_4bit": False,
        "load_in_8bit": precision == "8bit",
        "load_in_16bit": precision == "bf16",
        "dtype": torch.bfloat16 if precision == "bf16" else None,
        "use_gradient_checkpointing": False,
        "device_map": "auto",
        "fix_tokenizer": False,
    }

    try:
        # Give Unsloth the adapter itself. Its loader resolves the base model,
        # attaches PEFT, then reapplies its VLM inference patches to the final
        # wrapped model. Loading the base first and wrapping it ourselves skips
        # that last post-patch phase.
        model, _ = FastVisionModel.from_pretrained(
            model_name=adapter_source,
            **load_kwargs,
        )
        logger.info("[Qari OCR] Adapter loaded through Unsloth's native PEFT path.")
    except Exception:
        # Keep a compatibility fallback for future Unsloth/PEFT changes.
        logger.warning(
            "[Qari OCR] Native adapter loading failed; using the compatible PEFT fallback.",
            exc_info=True,
        )
        from peft import PeftModel

        base, _ = FastVisionModel.from_pretrained(
            model_name=_cached_snapshot(BASE_MODEL),
            **load_kwargs,
        )
        model = PeftModel.from_pretrained(base, adapter_source)

    if not getattr(model, "peft_config", None):
        raise RuntimeError("L'adaptateur PEFT Qari n'a pas été appliqué.")
    FastVisionModel.for_inference(model)
    if not hasattr(model, "fast_generate"):
        raise RuntimeError("Les optimisations d'inférence Unsloth ne sont pas actives.")
    return model


def _init_model(progress=None):
    global _model, _processor
    if _model is not None and _processor is not None:
        return

    configure_transformers_compat()
    FastVisionModel = initialize_unsloth_runtime()
    configure_huggingface_http()
    import torch
    from transformers import AutoProcessor

    if progress:
        progress(0.1, desc="Chargement du processeur Qari OCR 0.4.0…")
    _processor = _load_processor(AutoProcessor)
    if progress:
        progress(0.3, desc="Chargement de Qari OCR 0.4.0 — Unsloth 8 bits…")
    try:
        _model = _load_model(FastVisionModel, torch)
        _model.eval()
    except Exception:
        _model = None
        _processor = None
        release_model(None)
        raise
    generate_impl = getattr(getattr(_model, "generate", None), "__name__", "unknown")
    has_vllm = getattr(_model, "vllm_engine", None) is not None
    logger.info(
        "[Qari OCR] Unsloth model ready; precision=%s; generate=%s; vllm=%s; prompt=%r.",
        _inference_precision(),
        generate_impl,
        has_vllm,
        f"{OCR_PROMPT}.",
    )
    if progress:
        progress(1.0, desc="Qari prêt")


def _crop_white_page_margins(image):
    """Remove blank page margins while keeping the source text resolution."""
    if os.getenv("QARI_AUTO_CROP", "1").strip().lower() in {"0", "false", "no"}:
        return image, image.size

    threshold = max(0, min(255, int(os.getenv("QARI_CROP_WHITE_THRESHOLD", "245"))))
    margin = max(0, int(os.getenv("QARI_CROP_MARGIN_PIXELS", "64")))
    gray = image.convert("L")
    ink_mask = gray.point(lambda value: 255 if value < threshold else 0)
    bbox = ink_mask.getbbox()
    if bbox is None:
        return image, image.size

    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(image.width, right + margin)
    bottom = min(image.height, bottom + margin)
    cropped_size = (right - left, bottom - top)
    if cropped_size[0] * cropped_size[1] >= image.width * image.height * 0.95:
        return image, image.size
    return image.crop((left, top, right, bottom)), cropped_size


def _run_qari_page(image_path: str, progress_cb=None) -> str:
    """Transcribe one image using Qari's official multimodal chat format."""
    import torch
    from PIL import Image
    from transformers import StoppingCriteria, StoppingCriteriaList

    if _model is None or _processor is None:
        raise RuntimeError("Qari n'est pas chargé.")

    started_at = time.perf_counter()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        original_size = image.size
        image, cropped_size = _crop_white_page_margins(image)
        if cropped_size != original_size:
            logger.info(
                "[Qari OCR] Blank margins cropped: %sx%s -> %sx%s pixels.",
                original_size[0],
                original_size[1],
                cropped_size[0],
                cropped_size[1],
            )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"{OCR_PROMPT}."},
            ],
        }]
        inputs = _processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"truncation": False, "padding": False},
        )
        inputs = inputs.to(model_device(_model))
        prepared_at = time.perf_counter()
        input_tokens = int(inputs["input_ids"].shape[-1])
        if progress_cb:
            progress_cb(
                f"image préparée en {prepared_at - started_at:.1f}s "
                f"({cropped_size[0]}×{cropped_size[1]}, {input_tokens:,} jetons), "
                "génération en cours…"
            )

        class _GenerationHeartbeat(StoppingCriteria):
            """Report autoregressive progress without changing when generation stops."""

            def __init__(self):
                self.last_report_at = prepared_at
                self.interval = float(os.getenv("QARI_PROGRESS_INTERVAL_SECONDS", "15"))

            def __call__(self, input_ids, scores, **kwargs):
                now = time.perf_counter()
                if now - self.last_report_at < self.interval:
                    return False

                generated_tokens = max(0, int(input_ids.shape[-1]) - input_tokens)
                elapsed = now - prepared_at
                speed = generated_tokens / elapsed if elapsed else 0.0
                message = (
                    f"{generated_tokens:,} jetons générés en {elapsed:.0f}s "
                    f"({speed:.1f} jeton/s), toujours en cours…"
                )
                logger.info("[Qari OCR] Generation progress: %s", message)
                if progress_cb:
                    progress_cb(message)
                self.last_report_at = now
                return False

        max_new_tokens = int(os.getenv("QARI_MAX_NEW_TOKENS", "2048"))
        logger.info(
            "[Qari OCR] Generation started: backend=unsloth input_tokens=%s "
            "max_new_tokens=%s precision=%s",
            input_tokens,
            max_new_tokens,
            _inference_precision(),
        )

        with torch.inference_mode():
            generated = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([_GenerationHeartbeat()]),
            )

    finished_at = time.perf_counter()
    output_tokens = int(generated.shape[-1] - input_tokens)
    logger.info(
        "[Qari OCR] Page complete: backend=unsloth input_tokens=%s output_tokens=%s "
        "prepare=%.1fs generate=%.1fs total=%.1fs precision=%s",
        input_tokens,
        output_tokens,
        prepared_at - started_at,
        finished_at - prepared_at,
        finished_at - started_at,
        _inference_precision(),
    )
    if progress_cb:
        progress_cb(
            f"terminée en {finished_at - started_at:.1f}s "
            f"({output_tokens} jetons générés)."
        )

    prompt_length = inputs["input_ids"].shape[-1]
    result = _processor.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return clean_ocr_text(result)


def load_parsed_ocr(pdf_path):
    return load_page_map(get_ocr_text_path(pdf_path))


def save_parsed_ocr(pdf_path, pages_dict):
    return save_page_map(get_ocr_text_path(pdf_path), pages_dict)


def get_page_text(pdf_path, page_num):
    return load_parsed_ocr(pdf_path).get(int(page_num), "[Non transcrit]")


def run_ocr_batch(pdf_path, start_page, end_page, progress_cb=None):
    if not pdf_path or not os.path.exists(pdf_path):
        return "Erreur : PDF introuvable."
    try:
        import fitz

        with fitz.open(pdf_path) as document:
            total = len(document)
        start, end = validate_page_range(start_page, end_page, total)
        if end - start + 1 > 15:
            return "Erreur : maximum 15 pages par lot."

        _init_model()
        images_dir = os.path.join(get_pdf_workspace_dir(pdf_path), "images")
        pages = load_parsed_ocr(pdf_path)
        for page in range(start, end + 1):
            image_path = os.path.join(images_dir, f"page_{page:04d}.png")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image de la page {page} introuvable.")
            if progress_cb:
                progress_cb(f"Qari — page {page}/{end}…\n")
            page_progress = None
            if progress_cb:
                page_progress = lambda message, current=page: progress_cb(
                    f"Qari — page {current}: {message}\n"
                )
            pages[page] = _run_qari_page(image_path, page_progress)
            save_parsed_ocr(pdf_path, pages)
            if progress_cb:
                progress_cb(f"Qari — page {page} sauvegardée.\n")
        return f"Qari terminé : pages {start}–{end}."
    except Exception as exc:
        logger.exception("[Qari OCR] Batch failed")
        return f"Échec Qari OCR : {exc}"
