"""Shared helpers for the Arabic vision OCR backends."""

from __future__ import annotations

import gc
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from modules.utils import logger


OCR_PROMPT = "Free OCR"
DEFAULT_DPI = 300
OCR_CONTEXT_LENGTH = 262_144
_page_map_locks: dict[str, threading.Lock] = {}
_page_map_locks_guard = threading.Lock()


def disable_hidden_tokenizer_truncation(processor: Any) -> Any:
    """Disable truncation persisted inside some adapter tokenizer.json files.

    Some OCR adapters declare a 262k context but their tokenizer backend is
    saved with a 5,120-token truncation rule. On a full 300 DPI page this silently
    removes image placeholders and the multimodal processor then rejects the
    mismatched input. Explicitly clearing the backend rule is required; merely
    setting ``model_max_length`` is not enough.
    """
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return processor

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "no_truncation"):
        backend.no_truncation()

    current_limit = getattr(tokenizer, "model_max_length", 0)
    if not isinstance(current_limit, int) or current_limit < OCR_CONTEXT_LENGTH:
        tokenizer.model_max_length = OCR_CONTEXT_LENGTH
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        init_kwargs["model_max_length"] = tokenizer.model_max_length

    logger.info(
        "[OCR] Tokenizer truncation disabled (context=%s).",
        tokenizer.model_max_length,
    )
    return processor


def configure_transformers_compat() -> None:
    """Bridge torch 2.6 with optional torchao releases requiring torch 2.11."""
    import torch

    pytree = torch.utils._pytree
    if hasattr(pytree, "register_constant"):
        return

    def register_constant(cls):
        try:
            pytree.register_pytree_node(
                cls,
                lambda instance: ([], instance),
                lambda _children, instance: instance,
            )
        except ValueError:
            # The class may already have been registered by another backend.
            pass
        return cls

    pytree.register_constant = register_constant
    logger.info("[OCR] Installed torch pytree compatibility shim for torchao.")


def configure_huggingface_http() -> None:
    """Keep the existing local certificate workaround idempotent."""
    import httpx
    import requests
    import urllib3

    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(project_root / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if getattr(requests.Session.send, "_alkutub_noverify", False):
        return

    original_send = requests.Session.send

    def send_without_verification(self, request, **kwargs):
        kwargs["verify"] = False
        return original_send(self, request, **kwargs)

    send_without_verification._alkutub_noverify = True
    requests.Session.send = send_without_verification

    try:
        from huggingface_hub import configure_http_backend

        def backend_factory() -> requests.Session:
            session = requests.Session()
            session.verify = False
            return session

        configure_http_backend(backend_factory=backend_factory)
    except Exception as exc:
        logger.debug("[OCR] Hugging Face HTTP backend was not replaced: %s", exc)

    # huggingface_hub >= 1.0 uses httpx instead of requests.
    try:
        from huggingface_hub import set_async_client_factory, set_client_factory
        from huggingface_hub.utils._http import (
            async_hf_request_event_hook,
            async_hf_response_event_hook,
            hf_request_event_hook,
        )

        set_client_factory(lambda: httpx.Client(
            verify=False,
            event_hooks={"request": [hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
        ))
        set_async_client_factory(lambda: httpx.AsyncClient(
            verify=False,
            event_hooks={
                "request": [async_hf_request_event_hook],
                "response": [async_hf_response_event_hook],
            },
            follow_redirects=True,
            timeout=None,
        ))
    except Exception as exc:
        logger.debug("[OCR] Hugging Face httpx backend was not replaced: %s", exc)


def preferred_dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def model_device(model):
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def release_model(model: Any) -> None:
    """Release a model without moving quantized weights back through host RAM."""
    if model is not None:
        del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def validate_page_range(start_page: Any, end_page: Any, total_pages: int | None = None):
    try:
        start = int(start_page)
        end = int(end_page)
    except (TypeError, ValueError) as exc:
        raise ValueError("Les numéros de page doivent être des entiers.") from exc
    if start < 1 or end < start:
        raise ValueError("La plage de pages est invalide.")
    if total_pages is not None and end > total_pages:
        raise ValueError(f"Le PDF contient seulement {total_pages} pages.")
    return start, end


def clean_generated_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.DOTALL)
    for token in ("<|im_end|>", "<|endoftext|>", "<PAGE>"):
        text = text.replace(token, "")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _parse_page_map_file(path: Path) -> dict[int, str]:
    pages: dict[int, str] = {}
    try:
        content = path.read_text(encoding="utf-8")
        for part in content.split("--- PAGE ")[1:]:
            header, separator, body = part.partition("\n")
            if not separator:
                continue
            match = re.match(r"\s*(\d+)\s*---", header)
            if match:
                pages[int(match.group(1))] = body.strip("\r\n")
    except Exception as exc:
        logger.error("[OCR] Could not parse %s: %s", path, exc)
    return pages


def load_page_map(path: str) -> dict[int, str]:
    destination = Path(path)
    candidates = [destination]
    legacy_temporary = destination.with_suffix(destination.suffix + ".tmp")
    if legacy_temporary.exists():
        candidates.append(legacy_temporary)
    candidates.extend(
        candidate
        for candidate in destination.parent.glob(f".{destination.name}.*.tmp")
        if candidate.is_file()
    )
    existing = sorted(
        {candidate.resolve() for candidate in candidates if candidate.exists()},
        key=lambda candidate: candidate.stat().st_mtime_ns,
    )
    pages: dict[int, str] = {}
    for candidate in existing:
        pages.update(_parse_page_map_file(candidate))
        if candidate != destination.resolve():
            logger.warning(
                "[OCR] Recovered pending page-map data from %s.",
                candidate,
            )
    return pages


def save_page_map(path: str, pages: dict[int, str]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n".join(
        f"--- PAGE {page} ---\n{pages[page]}" for page in sorted(pages)
    )

    lock_key = str(destination.resolve()).casefold()
    with _page_map_locks_guard:
        destination_lock = _page_map_locks.setdefault(
            lock_key,
            threading.Lock(),
        )

    with destination_lock:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                delete=False,
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=f".{os.getpid()}.{threading.get_ident()}.tmp",
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)

            retry_seconds = max(
                0.0,
                float(os.getenv("OCR_SAVE_RETRY_SECONDS", "10")),
            )
            deadline = time.monotonic() + retry_seconds
            attempt = 0
            while True:
                try:
                    os.replace(temporary_path, destination)
                    temporary_path = None
                    stale_recovery_files = [
                        destination.with_suffix(destination.suffix + ".tmp"),
                        *destination.parent.glob(
                            f".{destination.name}.*.tmp"
                        ),
                    ]
                    for recovery_file in stale_recovery_files:
                        if recovery_file.exists():
                            try:
                                recovery_file.unlink()
                            except OSError:
                                logger.warning(
                                    "[OCR] Could not remove stale recovery "
                                    "file %s.",
                                    recovery_file,
                                )
                    break
                except PermissionError as exc:
                    if time.monotonic() >= deadline:
                        raise PermissionError(
                            f"{destination} reste verrouillé après "
                            f"{retry_seconds:.1f}s. Copie de récupération : "
                            f"{temporary_path}"
                        ) from exc
                    attempt += 1
                    delay = min(0.5, 0.05 * (2 ** min(attempt, 4)))
                    logger.warning(
                        "[OCR] %s is temporarily locked; retrying save in %.2fs.",
                        destination,
                        delay,
                    )
                    time.sleep(delay)
        finally:
            if temporary_path is not None and temporary_path.exists():
                logger.error(
                    "[OCR] Save failed; recovery copy preserved at %s.",
                    temporary_path,
                )
    return str(destination)


def render_pdf_cache(pdf_path: str, workspace_dir: str, progress=None):
    """Copy a PDF and render a DPI-aware, resumable image cache."""
    import fitz
    import shutil

    if not pdf_path or not os.path.exists(pdf_path):
        return None, 0

    source = Path(pdf_path).resolve()
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    destination = workspace / f"{source.stem}.pdf"
    if source != destination.resolve():
        shutil.copy2(source, destination)

    dpi = max(150, int(os.getenv("OCR_RENDER_DPI", str(DEFAULT_DPI))))
    images_dir = workspace / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = images_dir / "cache.json"

    with fitz.open(destination) as document:
        total = len(document)
        signature = {
            "dpi": dpi,
            "pages": total,
            "pdf_size": destination.stat().st_size,
            "pdf_mtime_ns": destination.stat().st_mtime_ns,
        }
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}

        regenerate_all = previous != signature
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        for index in range(total):
            image_path = images_dir / f"page_{index + 1:04d}.png"
            if regenerate_all or not image_path.exists():
                if progress:
                    progress(index / max(total, 1), desc=f"Préparation page {index + 1}/{total} à {dpi} DPI")
                pixmap = document[index].get_pixmap(matrix=matrix, alpha=False)
                pixmap.save(image_path)

    manifest_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return str(destination), total
