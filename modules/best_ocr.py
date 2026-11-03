"""GPU OCR using the local best checkpoint fine-tuned from Surya OCR 2."""

from __future__ import annotations

import html
import base64
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path

import fitz

from modules.ocr_common import (
    configure_huggingface_http,
    configure_transformers_compat,
    disable_hidden_tokenizer_truncation,
    load_page_map,
    model_device,
    release_model,
    save_page_map,
    validate_page_range,
)
from modules.utils import OUTPUTS_DIR, logger


BASE_MODEL_ID = "datalab-to/surya-ocr-2"
MODEL_LABEL = "Surya OCR 2 · modèle final SFT 9750"
DEFAULT_MAX_IMAGE_WIDTH = 1920
DEFAULT_VLLM_URL = "http://127.0.0.1:8012/v1"
BASE_VLLM_MODEL = "surya-base"
FINAL_VLLM_MODEL = "surya-final"
DEFAULT_VLLM_MODEL = FINAL_VLLM_MODEL
DEFAULT_WSL_DISTRO = "Ubuntu"
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_PARALLEL_PAGES = 8
OFFICIAL_FULL_PAGE_PROMPT = (
    "OCR this image to HTML. Each block is a div with data-label and data-bbox "
    "(x0 y0 x1 y1, normalized 0-1000)."
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "surya_sft_9750" / "SFT_9750_SAFE"
DEFAULT_CHECKPOINT_ZIP = Path.home() / "Downloads" / "MODELE_OCR_ARABE_SFT_9750.zip"
OCR_RUNS_DIR = Path(OUTPUTS_DIR) / "ocr_runs"
MODEL_VARIANTS = {
    "final": {
        "label": "Surya 2 · SFT 9750 final",
        "vllm_model": FINAL_VLLM_MODEL,
        "suffix": "surya_sft_9750",
    },
    "base": {
        "label": "Surya 2 · base",
        "vllm_model": BASE_VLLM_MODEL,
        "suffix": "surya_base",
    },
}
REQUIRED_CHECKPOINT_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
}
OPTIONAL_CHECKPOINT_FILES = {
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
}

_model = None
_processor = None
_model_lock = threading.Lock()
_batch_lock = threading.Lock()
_vllm_process = None


def _emit(callback, **event):
    if callback:
        callback(event)


def _checkpoint_zip_path() -> Path:
    configured = os.getenv("BEST_OCR_CHECKPOINT_ZIP", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_CHECKPOINT_ZIP


def ensure_checkpoint() -> Path:
    """Install only inference artifacts from the supplied checkpoint archive."""
    present = {path.name for path in CHECKPOINT_DIR.glob("*") if path.is_file()}
    if REQUIRED_CHECKPOINT_FILES.issubset(present):
        return CHECKPOINT_DIR

    archive_path = _checkpoint_zip_path()
    if not archive_path.exists():
        missing = ", ".join(sorted(REQUIRED_CHECKPOINT_FILES - present))
        raise FileNotFoundError(
            "Checkpoint Best OCR incomplet. "
            f"Fichiers manquants : {missing}. Archive attendue : {archive_path}"
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        by_name = {
            Path(entry.filename).name: entry
            for entry in archive.infolist()
            if not entry.is_dir()
        }
        missing = REQUIRED_CHECKPOINT_FILES - by_name.keys()
        if missing:
            raise RuntimeError(
                "Archive Best OCR invalide. Fichiers manquants : "
                + ", ".join(sorted(missing))
            )
        for name in REQUIRED_CHECKPOINT_FILES | OPTIONAL_CHECKPOINT_FILES:
            if name not in by_name:
                continue
            source = archive.open(by_name[name])
            try:
                with open(CHECKPOINT_DIR / name, "wb") as target:
                    shutil.copyfileobj(source, target)
            finally:
                source.close()
    return CHECKPOINT_DIR


def get_pdf_workspace_dir(pdf_path: str) -> str:
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    return str(OCR_RUNS_DIR / pdf_name)


def _model_variant(model_variant: str = "final") -> dict:
    try:
        return MODEL_VARIANTS[model_variant]
    except KeyError as exc:
        raise ValueError(f"Variante OCR inconnue : {model_variant}") from exc


def get_ocr_text_path(pdf_path: str, model_variant: str = "final") -> str:
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    suffix = _model_variant(model_variant)["suffix"]
    return str(Path(get_pdf_workspace_dir(pdf_path)) / f"{pdf_name}_{suffix}.txt")


def get_raw_html_dir(pdf_path: str, model_variant: str = "final") -> Path:
    suffix = _model_variant(model_variant)["suffix"]
    return Path(get_pdf_workspace_dir(pdf_path)) / f"{suffix}_html"


def load_parsed_ocr(pdf_path: str, model_variant: str = "final") -> dict[int, str]:
    return load_page_map(get_ocr_text_path(pdf_path, model_variant))


def save_parsed_ocr(
    pdf_path: str,
    pages_dict: dict[int, str],
    model_variant: str = "final",
) -> str:
    return save_page_map(get_ocr_text_path(pdf_path, model_variant), pages_dict)


def get_page_text(
    pdf_path: str,
    page_num: int,
    model_variant: str = "final",
) -> str:
    return load_parsed_ocr(pdf_path, model_variant).get(
        int(page_num),
        "[Non transcrit]",
    )


def get_pdf_total_pages(pdf_path: str) -> int:
    with fitz.open(pdf_path) as document:
        return len(document)


class _PlainTextHTMLParser(HTMLParser):
    BLOCK_TAGS = {
        "address",
        "article",
        "body",
        "blockquote",
        "br",
        "caption",
        "column",
        "columns",
        "course",
        "div",
        "figcaption",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "page",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def _newline(self):
        if self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.BLOCK_TAGS:
            self._newline()

    def handle_endtag(self, tag):
        if tag.lower() in self.BLOCK_TAGS:
            self._newline()

    def handle_data(self, data):
        if data:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def clean_html_ocr(raw_text: str) -> str:
    text = (raw_text or "").strip()
    text = re.sub(r"^\s*```(?:html)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    parser = _PlainTextHTMLParser()
    try:
        parser.feed(text)
        text = parser.text()
    except Exception:
        text = re.sub(r"<[^>]+>", "\n", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def scale_to_surya(image, max_size=(3072, 2048), min_size=(1792, 28), grid_size=28):
    """Mirror Surya OCR 2's official image scaling."""
    from PIL import Image

    width, height = image.size
    if width <= 0 or height <= 0:
        return image

    original_aspect = width / height
    current_pixels = width * height
    max_pixels = max_size[0] * max_size[1]
    min_pixels = min_size[0] * min_size[1]
    scale = 1.0
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
    elif current_pixels < min_pixels:
        scale = (min_pixels / current_pixels) ** 0.5

    width_blocks = max(1, round((width * scale) / grid_size))
    height_blocks = max(1, round((height * scale) / grid_size))
    while width_blocks * height_blocks * grid_size * grid_size > max_pixels:
        if width_blocks == 1 and height_blocks == 1:
            break
        if width_blocks == 1:
            height_blocks -= 1
            continue
        if height_blocks == 1:
            width_blocks -= 1
            continue
        width_loss = abs(((width_blocks - 1) / height_blocks) - original_aspect)
        height_loss = abs((width_blocks / (height_blocks - 1)) - original_aspect)
        if width_loss < height_loss:
            width_blocks -= 1
        else:
            height_blocks -= 1

    new_size = (width_blocks * grid_size, height_blocks * grid_size)
    if new_size == image.size:
        return image
    return image.resize(new_size, resample=Image.Resampling.LANCZOS)


def prepare_surya_page(image):
    """Preserve page detail while respecting the configured width cap."""
    from PIL import Image

    max_width = max(
        28,
        int(os.getenv("BEST_OCR_MAX_IMAGE_WIDTH", str(DEFAULT_MAX_IMAGE_WIDTH))),
    )
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (
            max_width,
            max(28, round(image.height * ratio)),
        )
        image = image.resize(new_size, resample=Image.Resampling.LANCZOS)
    image = scale_to_surya(image)
    if image.width > max_width:
        width = max(28, (max_width // 28) * 28)
        height = max(28, round((image.height * width / image.width) / 28) * 28)
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
    return image


def gpu_snapshot() -> dict:
    snapshot = {
        "gpu_util": 0,
        "vram_used_mb": 0,
        "vram_total_mb": 0,
        "temperature_c": 0,
        "power_w": 0.0,
    }
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
            creationflags=flags,
        )
        values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]
        snapshot.update(
            gpu_util=int(float(values[0])),
            vram_used_mb=int(float(values[1])),
            vram_total_mb=int(float(values[2])),
            temperature_c=int(float(values[3])),
            power_w=float(values[4]),
        )
    except Exception as exc:
        logger.debug("[Best OCR] GPU telemetry unavailable: %s", exc)
    return snapshot


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "Calibration…"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {secs:02d} s"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def release_best_ocr_model():
    global _model, _processor
    model = _model
    _model = None
    _processor = None
    release_model(model)
    logger.info("[Best OCR] Model released.")


def stop_vllm_server() -> bool:
    """Stop the dedicated OCR WSL runtime and release its CUDA allocation."""
    global _vllm_process
    distro = os.getenv("BEST_OCR_WSL_DISTRO", DEFAULT_WSL_DISTRO)
    try:
        result = subprocess.run(
            ["wsl.exe", "--terminate", distro],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        _vllm_process = None
        if result.returncode not in (0, 1):
            logger.warning(
                "[Best OCR] WSL termination returned %s: %s",
                result.returncode,
                (result.stderr or result.stdout).strip(),
            )
            return False
        logger.info("[Best OCR] vLLM/WSL runtime stopped.")
        return True
    except FileNotFoundError:
        return False
    except Exception:
        logger.warning("[Best OCR] Could not stop vLLM/WSL.", exc_info=True)
        return False


def _backend_name() -> str:
    return os.getenv("BEST_OCR_BACKEND", "vllm").strip().lower()


def _vllm_url() -> str:
    return os.getenv("BEST_OCR_VLLM_URL", DEFAULT_VLLM_URL).rstrip("/")


def _vllm_is_ready() -> bool:
    import requests

    try:
        response = requests.get(
            _vllm_url().removesuffix("/v1") + "/health",
            timeout=2,
        )
        return response.ok
    except requests.RequestException:
        return False


def _vllm_model_names() -> set[str]:
    import requests

    try:
        response = requests.get(_vllm_url() + "/models", timeout=3)
        response.raise_for_status()
        return {
            str(model.get("id"))
            for model in (response.json().get("data") or [])
            if model.get("id")
        }
    except (requests.RequestException, ValueError):
        return set()


def _vllm_has_comparison_models() -> bool:
    names = _vllm_model_names()
    return {BASE_VLLM_MODEL, FINAL_VLLM_MODEL}.issubset(names)


def _to_wsl_path(path: Path) -> str:
    resolved = str(path.resolve())
    drive, tail = os.path.splitdrive(resolved)
    if not drive:
        return resolved.replace("\\", "/")
    normalized_tail = tail.lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive[0].lower()}/{normalized_tail}"


def _vllm_log_tail(log_path: Path, limit: int = 2500) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def _ensure_vllm_server(checkpoint: Path, progress_cb=None) -> None:
    """Start the official Surya vLLM GPU backend inside WSL when needed."""
    global _vllm_process

    if _vllm_is_ready():
        if _vllm_has_comparison_models():
            return
        logger.info(
            "[Best OCR] Existing vLLM server does not expose %s and %s; restarting.",
            BASE_VLLM_MODEL,
            FINAL_VLLM_MODEL,
        )
        stop_vllm_server()

    distro = os.getenv("BEST_OCR_WSL_DISTRO", DEFAULT_WSL_DISTRO)
    vllm_bin = os.getenv(
        "BEST_OCR_VLLM_BIN",
        "~/.venvs/al-kutub-vllm/bin/vllm",
    )
    port = _vllm_url().split(":")[-1].split("/")[0]
    log_path = Path(OUTPUTS_DIR) / "best_ocr_vllm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_wsl = _to_wsl_path(checkpoint)
    gpu_memory = os.getenv("BEST_OCR_GPU_MEMORY_UTILIZATION", "0.67")
    max_model_len = os.getenv("BEST_OCR_MAX_MODEL_LEN", "13000")
    max_num_seqs = os.getenv("BEST_OCR_MAX_NUM_SEQS", "8")
    command = " ".join(
        [
            "exec",
            vllm_bin,
            "serve",
            shlex.quote(BASE_MODEL_ID),
            "--host 0.0.0.0",
            f"--port {shlex.quote(port)}",
            "--dtype bfloat16",
            f"--gpu-memory-utilization {shlex.quote(gpu_memory)}",
            f"--max-model-len {shlex.quote(max_model_len)}",
            f"--max-num-seqs {shlex.quote(max_num_seqs)}",
            "--limit-mm-per-prompt.image 1",
            "--limit-mm-per-prompt.video 0",
            "--skip-mm-profiling",
            "--enable-lora",
            "--max-lora-rank 16",
            "--enable-tower-connector-lora",
            "--served-model-name",
            BASE_VLLM_MODEL,
            "--lora-modules",
            shlex.quote(f"{FINAL_VLLM_MODEL}={checkpoint_wsl}"),
        ]
    )
    if os.getenv("BEST_OCR_FAST_STARTUP", "0") == "1":
        command += " --enforce-eager"
    _emit(
        progress_cb,
        type="telemetry",
        stage="loading",
        message=(
            "Démarrage du moteur GPU vLLM… "
            "Le tout premier lancement compile les accélérations."
        ),
        **gpu_snapshot(),
    )
    with open(log_path, "w", encoding="utf-8") as log_file:
        _vllm_process = subprocess.Popen(
            ["wsl", "-d", distro, "--", "bash", "-lc", command],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    timeout_seconds = float(os.getenv("BEST_OCR_VLLM_START_TIMEOUT", "420"))
    started = time.perf_counter()
    last_update = 0.0
    while time.perf_counter() - started < timeout_seconds:
        if _vllm_is_ready() and _vllm_has_comparison_models():
            logger.info(
                "[Best OCR] vLLM ready in %.1fs at %s",
                time.perf_counter() - started,
                _vllm_url(),
            )
            return
        if _vllm_process.poll() is not None:
            details = _vllm_log_tail(log_path)
            raise RuntimeError(
                "Le moteur vLLM n’a pas pu démarrer. "
                f"Consultez {log_path}.\n{details}"
            )
        elapsed = time.perf_counter() - started
        if elapsed - last_update >= 2:
            _emit(
                progress_cb,
                type="telemetry",
                stage="loading",
                message=f"Initialisation vLLM : {format_duration(elapsed)}…",
                elapsed_seconds=elapsed,
                **gpu_snapshot(),
            )
            last_update = elapsed
        time.sleep(0.5)
    raise TimeoutError(
        f"vLLM ne répond pas après {format_duration(timeout_seconds)}. "
        f"Consultez {log_path}."
    )


def _load_processor(checkpoint: Path):
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(
            str(checkpoint),
            local_files_only=True,
        )
    except Exception as exc:
        logger.warning(
            "[Best OCR] Local adapter processor unavailable; using Surya base: %s",
            exc,
        )
        processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    return disable_hidden_tokenizer_truncation(processor)


def _load_model(checkpoint: Path):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Best OCR nécessite un GPU NVIDIA CUDA.")

    from modules.qari_ocr import initialize_unsloth_runtime

    FastVisionModel = initialize_unsloth_runtime()
    model, _ = FastVisionModel.from_pretrained(
        model_name=str(checkpoint),
        tokenizer_name=str(checkpoint),
        max_seq_length=int(os.getenv("BEST_OCR_MAX_SEQ_LENGTH", "18000")),
        dtype=torch.bfloat16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        use_gradient_checkpointing=False,
        device_map="auto",
        fix_tokenizer=False,
    )
    if not getattr(model, "peft_config", None):
        raise RuntimeError("L’adaptateur Best OCR n’a pas été appliqué.")
    FastVisionModel.for_inference(model)
    return model


def _init_model(progress_cb=None):
    global _model, _processor
    if _backend_name() == "vllm":
        checkpoint = ensure_checkpoint()
        release_best_ocr_model()
        started = time.perf_counter()
        _ensure_vllm_server(checkpoint, progress_cb)
        _emit(
            progress_cb,
            type="telemetry",
            stage="ready",
            message=(
                f"Surya fine-tuné prêt avec vLLM en "
                f"{format_duration(time.perf_counter() - started)}."
            ),
            elapsed_seconds=time.perf_counter() - started,
            **gpu_snapshot(),
        )
        return

    if _model is not None and _processor is not None:
        return

    with _model_lock:
        if _model is not None and _processor is not None:
            return
        checkpoint = ensure_checkpoint()
        configure_transformers_compat()
        configure_huggingface_http()

        # Unsloth must patch Transformers before AutoProcessor imports model code.
        from modules.qari_ocr import initialize_unsloth_runtime

        initialize_unsloth_runtime()
        _emit(
            progress_cb,
            type="telemetry",
            stage="loading",
            message="Chargement de Surya fine-tuné sur le GPU…",
            **gpu_snapshot(),
        )

        try:
            from modules.qari_ocr import release_qari_model

            release_qari_model()
        except Exception:
            logger.debug("[Best OCR] Qari was not loaded.", exc_info=True)

        started = time.perf_counter()
        try:
            _processor = _load_processor(checkpoint)
            _model = _load_model(checkpoint)
            _model.eval()
        except Exception:
            _model = None
            _processor = None
            release_model(None)
            raise
        load_seconds = time.perf_counter() - started
        logger.info(
            "[Best OCR] Model ready: base=%s checkpoint=%s load=%.1fs device=%s",
            BASE_MODEL_ID,
            checkpoint,
            load_seconds,
            model_device(_model),
        )
        _emit(
            progress_cb,
            type="telemetry",
            stage="ready",
            message=f"Surya fine-tuné chargé en {format_duration(load_seconds)}.",
            elapsed_seconds=load_seconds,
            **gpu_snapshot(),
        )


def _run_page_vllm(
    image_path: str,
    progress_cb=None,
    model_variant: str = "final",
    max_tokens: int | None = None,
) -> tuple[str, str, dict]:
    import requests
    from PIL import Image

    variant = _model_variant(model_variant)
    started = time.perf_counter()
    with Image.open(image_path) as source:
        original_size = source.size
        image = prepare_surya_page(source.convert("RGB"))
        encoded = io.BytesIO()
        image.save(encoded, format="PNG", optimize=True)
    prepared = time.perf_counter()

    requested_max_tokens = int(
        max_tokens
        or os.getenv("BEST_OCR_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS))
    )
    payload = {
        "model": variant["vllm_model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                + base64.b64encode(encoded.getvalue()).decode("ascii")
                            )
                        },
                    },
                    {"type": "text", "text": OFFICIAL_FULL_PAGE_PROMPT},
                ],
            }
        ],
        "max_tokens": requested_max_tokens,
        "temperature": 0,
    }
    _emit(
        progress_cb,
        type="page_progress",
        stage="generating",
        message=(
            f"Image {original_size[0]}×{original_size[1]} → "
            f"{image.width}×{image.height}; {variant['label']} avec vLLM."
        ),
        page_elapsed_seconds=prepared - started,
        input_tokens=0,
        generated_tokens=0,
        tokens_per_second=0.0,
        **gpu_snapshot(),
    )

    interval = max(0.5, float(os.getenv("BEST_OCR_TELEMETRY_INTERVAL", "2")))

    def request_completion(request_payload: dict) -> tuple[dict, float]:
        result: dict = {}

        def request_ocr():
            try:
                response = requests.post(
                    _vllm_url() + "/chat/completions",
                    json=request_payload,
                    timeout=float(
                        os.getenv("BEST_OCR_VLLM_REQUEST_TIMEOUT", "900")
                    ),
                )
                response.raise_for_status()
                result["response"] = response.json()
            except Exception as exc:
                result["error"] = exc

        request_started = time.perf_counter()
        worker = threading.Thread(target=request_ocr, daemon=True)
        worker.start()
        while worker.is_alive():
            worker.join(timeout=interval)
            if worker.is_alive():
                elapsed = time.perf_counter() - request_started
                _emit(
                    progress_cb,
                    type="page_progress",
                    stage="generating",
                    message=(
                        f"{variant['label']} transcrit la page depuis "
                        f"{format_duration(elapsed)}…"
                    ),
                    page_elapsed_seconds=time.perf_counter() - started,
                    input_tokens=0,
                    generated_tokens=0,
                    tokens_per_second=0.0,
                    **gpu_snapshot(),
                )
        if "error" in result:
            raise RuntimeError(
                f"Erreur du moteur vLLM : {result['error']}"
            ) from result["error"]
        return result["response"], time.perf_counter() - request_started

    generation_started = time.perf_counter()
    response_data, first_request_seconds = request_completion(payload)
    choices = response_data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Réponse vLLM vide : {json.dumps(response_data)[:1000]}")
    finish_reason = choices[0].get("finish_reason")
    retried = False
    fallback_model = None
    if (
        finish_reason == "length"
        and model_variant == "final"
        and os.getenv("BEST_OCR_BASE_FALLBACK_ON_LENGTH", "1") != "0"
    ):
        retried = True
        retry_payload = dict(payload)
        retry_payload["model"] = BASE_VLLM_MODEL
        retry_payload["max_tokens"] = DEFAULT_MAX_NEW_TOKENS
        fallback_model = "base"
        _emit(
            progress_cb,
            type="page_progress",
            stage="retrying",
            message=(
                f"{variant['label']} a atteint {requested_max_tokens:,} jetons "
                "sans s’arrêter (boucle probable) ; reprise de sécurité avec "
                "Surya base."
            ),
            page_elapsed_seconds=time.perf_counter() - started,
            input_tokens=0,
            generated_tokens=requested_max_tokens,
            tokens_per_second=0.0,
            **gpu_snapshot(),
        )
        response_data, retry_seconds = request_completion(retry_payload)
        first_request_seconds += retry_seconds
        choices = response_data.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"Réponse vLLM vide après relance : "
                f"{json.dumps(response_data)[:1000]}"
            )
        finish_reason = choices[0].get("finish_reason")

    finished = time.perf_counter()
    raw_html = (choices[0].get("message") or {}).get("content", "").strip()
    usage = response_data.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    generation_seconds = finished - generation_started
    plain_text = clean_html_ocr(raw_html)
    metrics = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prepare_seconds": generation_started - started,
        "generation_seconds": generation_seconds,
        "total_seconds": finished - started,
        "tokens_per_second": (
            output_tokens / generation_seconds if generation_seconds else 0.0
        ),
        "backend": "vllm",
        "model_variant": model_variant,
        "finish_reason": finish_reason,
        "retried": retried,
        "fallback_model": fallback_model,
        "request_seconds": first_request_seconds,
    }
    logger.info(
        "[Best OCR] vLLM page complete: variant=%s input=%s output=%s "
        "prepare=%.2fs generate=%.2fs total=%.2fs speed=%.1f tok/s "
        "finish=%s retried=%s fallback=%s",
        model_variant,
        input_tokens,
        output_tokens,
        metrics["prepare_seconds"],
        metrics["generation_seconds"],
        metrics["total_seconds"],
        metrics["tokens_per_second"],
        finish_reason,
        retried,
        fallback_model,
    )
    return plain_text, raw_html, metrics


def _run_page_transformers(image_path: str, progress_cb=None) -> tuple[str, str, dict]:
    import torch
    from PIL import Image
    from transformers import StoppingCriteria, StoppingCriteriaList

    if _model is None or _processor is None:
        raise RuntimeError("Best OCR n’est pas chargé.")

    started = time.perf_counter()
    with Image.open(image_path) as source:
        original_size = source.size
        image = prepare_surya_page(source.convert("RGB"))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": OFFICIAL_FULL_PAGE_PROMPT},
                ],
            }
        ]
        inputs = _processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"truncation": False, "padding": False},
        )
        inputs = inputs.to(model_device(_model))
        prepared = time.perf_counter()
        input_tokens = int(inputs["input_ids"].shape[-1])
        _emit(
            progress_cb,
            type="page_progress",
            stage="generating",
            message=(
                f"Image {original_size[0]}×{original_size[1]} → "
                f"{image.width}×{image.height} (détails préservés, "
                f"largeur plafonnée à "
                f"{os.getenv('BEST_OCR_MAX_IMAGE_WIDTH', str(DEFAULT_MAX_IMAGE_WIDTH))} px); "
                f"{input_tokens:,} jetons d’entrée."
            ),
            page_elapsed_seconds=prepared - started,
            input_tokens=input_tokens,
            generated_tokens=0,
            tokens_per_second=0.0,
            **gpu_snapshot(),
        )

        class _GenerationTelemetry(StoppingCriteria):
            def __init__(self):
                self.last_report = prepared
                self.interval = float(os.getenv("BEST_OCR_TELEMETRY_INTERVAL", "2"))

            def __call__(self, input_ids, scores, **kwargs):
                now = time.perf_counter()
                if now - self.last_report < self.interval:
                    return False
                generated_tokens = max(0, int(input_ids.shape[-1]) - input_tokens)
                elapsed = now - prepared
                speed = generated_tokens / elapsed if elapsed else 0.0
                _emit(
                    progress_cb,
                    type="page_progress",
                    stage="generating",
                    message=f"{generated_tokens:,} jetons générés à {speed:.1f} jetons/s.",
                    page_elapsed_seconds=now - started,
                    input_tokens=input_tokens,
                    generated_tokens=generated_tokens,
                    tokens_per_second=speed,
                    **gpu_snapshot(),
                )
                self.last_report = now
                return False

        max_new_tokens = int(
            os.getenv("BEST_OCR_MAX_NEW_TOKENS", str(DEFAULT_MAX_NEW_TOKENS))
        )
        torch.cuda.synchronize()
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([_GenerationTelemetry()]),
            )
        torch.cuda.synchronize()
        finished = time.perf_counter()

    output_tokens = int(generated.shape[-1] - input_tokens)
    raw_html = _processor.batch_decode(
        generated[:, input_tokens:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    plain_text = clean_html_ocr(raw_html)
    metrics = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prepare_seconds": generation_started - started,
        "generation_seconds": finished - generation_started,
        "total_seconds": finished - started,
        "tokens_per_second": (
            output_tokens / (finished - generation_started)
            if finished > generation_started
            else 0.0
        ),
    }
    logger.info(
        "[Best OCR] Page complete: input=%s output=%s prepare=%.2fs "
        "generate=%.2fs total=%.2fs speed=%.1f tok/s",
        input_tokens,
        output_tokens,
        metrics["prepare_seconds"],
        metrics["generation_seconds"],
        metrics["total_seconds"],
        metrics["tokens_per_second"],
    )
    return plain_text, raw_html, metrics


def _run_page(
    image_path: str,
    progress_cb=None,
    model_variant: str = "final",
) -> tuple[str, str, dict]:
    if _backend_name() == "vllm":
        return _run_page_vllm(
            image_path,
            progress_cb,
            model_variant=model_variant,
        )
    if model_variant != "final":
        raise RuntimeError(
            "La comparaison avec Surya base nécessite le backend vLLM."
        )
    return _run_page_transformers(image_path, progress_cb)


def run_ocr_batch(
    pdf_path,
    start_page,
    end_page,
    progress_cb=None,
    model_variant: str = "final",
    parallel_pages: int | None = None,
):
    if not pdf_path or not os.path.exists(pdf_path):
        return "Erreur : PDF introuvable."
    variant = _model_variant(model_variant)
    if not _batch_lock.acquire(blocking=False):
        return "Erreur : une tâche Best OCR est déjà en cours."

    try:
        total_document_pages = get_pdf_total_pages(pdf_path)
        start, end = validate_page_range(start_page, end_page, total_document_pages)
        selected_total = end - start + 1
        batch_started = time.perf_counter()
        page_durations: deque[float] = deque(maxlen=5)
        initial_page_estimate = max(
            1.0,
            float(os.getenv("BEST_OCR_INITIAL_PAGE_SECONDS", "12")),
        )
        pages = load_parsed_ocr(pdf_path, model_variant)
        images_dir = Path(get_pdf_workspace_dir(pdf_path)) / "images"
        raw_dir = get_raw_html_dir(pdf_path, model_variant)
        raw_dir.mkdir(parents=True, exist_ok=True)

        _init_model(progress_cb)
        configured_parallelism = (
            parallel_pages
            if parallel_pages is not None
            else int(
                os.getenv(
                    "BEST_OCR_PARALLEL_PAGES",
                    str(DEFAULT_PARALLEL_PAGES),
                )
            )
        )
        worker_count = (
            min(max(1, configured_parallelism), selected_total)
            if _backend_name() == "vllm"
            else 1
        )
        state_lock = threading.Lock()
        state = {"completed": 0}

        _emit(
            progress_cb,
            type="batch_start",
            stage="generating",
            message=(
                f"{variant['label']} : {selected_total} page(s), "
                f"{worker_count} page(s) en parallèle, "
                f"{DEFAULT_MAX_NEW_TOKENS:,} jetons maximum."
            ),
            engine=model_variant,
            total_pages=selected_total,
            completed_pages=0,
            remaining_pages=selected_total,
            parallel_pages=worker_count,
            elapsed_seconds=0,
            eta_seconds=(
                initial_page_estimate
                * ((selected_total + worker_count - 1) // worker_count)
            ),
            **gpu_snapshot(),
        )

        def run_one_page(page: int):
            image_path = images_dir / f"page_{page:04d}.png"
            if not image_path.exists():
                raise FileNotFoundError(f"Image de la page {page} introuvable.")

            with state_lock:
                completed_before = state["completed"]
                average = (
                    sum(page_durations) / len(page_durations)
                    if page_durations
                    else initial_page_estimate
                )
            remaining_before = selected_total - completed_before
            initial_eta = average * (
                (remaining_before + worker_count - 1) // worker_count
            )
            _emit(
                progress_cb,
                type="page_start",
                stage="generating",
                message=f"{variant['label']} · PAGE {page} en cours",
                engine=model_variant,
                page=page,
                page_index=completed_before + 1,
                total_pages=selected_total,
                completed_pages=completed_before,
                remaining_pages=remaining_before,
                elapsed_seconds=time.perf_counter() - batch_started,
                eta_seconds=initial_eta,
                completion_time=(
                    (datetime.now() + timedelta(seconds=initial_eta)).strftime("%H:%M:%S")
                    if initial_eta is not None
                    else None
                ),
                **gpu_snapshot(),
            )

            page_started = time.perf_counter()

            def page_progress(event):
                now = time.perf_counter()
                with state_lock:
                    completed_now = state["completed"]
                    average_now = (
                        sum(page_durations) / len(page_durations)
                        if page_durations
                        else initial_page_estimate
                    )
                remaining_now = selected_total - completed_now
                eta = average_now * (
                    (remaining_now + worker_count - 1) // worker_count
                )
                _emit(
                    progress_cb,
                    **event,
                    engine=model_variant,
                    page=page,
                    page_index=completed_now + 1,
                    total_pages=selected_total,
                    completed_pages=completed_now,
                    remaining_pages=remaining_now,
                    elapsed_seconds=now - batch_started,
                    eta_seconds=eta,
                    completion_time=(
                        (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M:%S")
                        if eta is not None
                        else None
                    ),
                )

            plain_text, raw_html, metrics = _run_page(
                str(image_path),
                page_progress,
                model_variant=model_variant,
            )
            page_duration = time.perf_counter() - page_started
            return page, plain_text, raw_html, metrics, page_duration

        page_numbers = list(range(start, end + 1))
        deferred_save_error: Exception | None = None
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=f"surya-{model_variant}",
        ) as executor:
            futures = {
                executor.submit(run_one_page, page): page
                for page in page_numbers
            }
            for future in as_completed(futures):
                page, plain_text, raw_html, metrics, page_duration = future.result()
                pages[page] = plain_text
                (raw_dir / f"page_{page:04d}.html").write_text(
                    raw_html,
                    encoding="utf-8",
                )
                (raw_dir / f"page_{page:04d}.txt").write_text(
                    plain_text,
                    encoding="utf-8",
                )
                page_save_deferred = False
                try:
                    save_parsed_ocr(pdf_path, pages, model_variant)
                    deferred_save_error = None
                except PermissionError as exc:
                    page_save_deferred = True
                    deferred_save_error = exc
                    logger.warning(
                        "[Best OCR] Aggregate output is locked; PAGE %s is "
                        "preserved separately and the aggregate save is deferred.",
                        page,
                    )

                with state_lock:
                    page_durations.append(page_duration)
                    state["completed"] += 1
                    completed = state["completed"]
                    rolling_average = sum(page_durations) / len(page_durations)

                remaining = selected_total - completed
                eta = rolling_average * (
                    (remaining + worker_count - 1) // worker_count
                )
                retry_note = (
                    " · boucle du modèle final remplacée par Surya base"
                    if metrics.get("fallback_model") == "base"
                    else ""
                )
                save_note = (
                    " · agrégat en attente (page sécurisée séparément)"
                    if page_save_deferred
                    else ""
                )
                _emit(
                    progress_cb,
                    type="page_complete",
                    stage="saving",
                    message=(
                        f"{variant['label']} · PAGE {page} sauvegardée en "
                        f"{format_duration(page_duration)} "
                        f"({metrics['output_tokens']:,} jetons)"
                        f"{retry_note}{save_note}."
                    ),
                    engine=model_variant,
                    page=page,
                    page_index=completed,
                    total_pages=selected_total,
                    completed_pages=completed,
                    remaining_pages=remaining,
                    elapsed_seconds=time.perf_counter() - batch_started,
                    page_seconds=page_duration,
                    average_page_seconds=rolling_average,
                    eta_seconds=eta,
                    completion_time=(
                        (datetime.now() + timedelta(seconds=eta)).strftime(
                            "%H:%M:%S"
                        )
                        if remaining
                        else datetime.now().strftime("%H:%M:%S")
                    ),
                    tokens_per_second=metrics["tokens_per_second"],
                    output_tokens=metrics["output_tokens"],
                    finish_reason=metrics.get("finish_reason"),
                    retried=metrics.get("retried", False),
                    fallback_model=metrics.get("fallback_model"),
                    **gpu_snapshot(),
                )

        if deferred_save_error is not None:
            logger.info(
                "[Best OCR] Retrying deferred aggregate save after all pages."
            )
        save_parsed_ocr(pdf_path, pages, model_variant)

        total_seconds = time.perf_counter() - batch_started
        _emit(
            progress_cb,
            type="complete",
            stage="complete",
            message=(
                f"{variant['label']} terminé : pages {start}–{end} en "
                f"{format_duration(total_seconds)}."
            ),
            engine=model_variant,
            page=end,
            page_index=selected_total,
            total_pages=selected_total,
            completed_pages=selected_total,
            remaining_pages=0,
            elapsed_seconds=total_seconds,
            eta_seconds=0,
            completion_time=datetime.now().strftime("%H:%M:%S"),
            **gpu_snapshot(),
        )
        return f"{variant['label']} terminé : pages {start}–{end}."
    except Exception as exc:
        logger.exception("[Best OCR] Batch failed (%s)", model_variant)
        _emit(
            progress_cb,
            type="error",
            stage="error",
            message=str(exc),
            engine=model_variant,
            **gpu_snapshot(),
        )
        return f"Échec {variant['label']} : {exc}"
    finally:
        _batch_lock.release()
