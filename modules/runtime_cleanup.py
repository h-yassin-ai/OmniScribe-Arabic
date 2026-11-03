"""Release local GPU runtimes when the Gradio application closes."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import threading

from modules.utils import logger


_cleanup_lock = threading.Lock()


def _call_release_if_loaded(module_name: str, function_name: str) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        return
    release = getattr(module, function_name, None)
    if callable(release):
        try:
            release()
        except Exception:
            logger.debug(
                "[Shutdown] Could not call %s.%s.",
                module_name,
                function_name,
                exc_info=True,
            )


def _release_unlimited_ocr_if_loaded() -> None:
    module = sys.modules.get("modules.unlimited_ocr")
    if module is None:
        return
    model = getattr(module, "_model", None)
    module._model = None
    module._tokenizer = None
    if model is not None:
        try:
            from modules.ocr_common import release_model

            release_model(model)
        except Exception:
            logger.debug(
                "[Shutdown] Could not release Unlimited OCR.",
                exc_info=True,
            )


def terminate_ocr_wsl() -> bool:
    """Terminate only the OCR Ubuntu distro, leaving Docker WSL untouched."""
    distro = os.getenv("BEST_OCR_WSL_DISTRO", "Ubuntu")
    try:
        result = subprocess.run(
            ["wsl.exe", "--terminate", distro],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode not in (0, 1):
            logger.warning(
                "[Shutdown] WSL termination returned %s: %s",
                result.returncode,
                (result.stderr or result.stdout).strip(),
            )
            return False
        logger.info("[Shutdown] WSL distro %s terminated.", distro)
        return True
    except FileNotFoundError:
        logger.info("[Shutdown] WSL is not installed.")
    except Exception:
        logger.warning("[Shutdown] Could not terminate WSL.", exc_info=True)
    return False


def cleanup_gpu_runtime() -> None:
    """Stop vLLM/WSL and release any model still held by the Windows process."""
    if not _cleanup_lock.acquire(blocking=False):
        return
    try:
        logger.info("[Shutdown] Releasing all OCR GPU resources…")
        terminate_ocr_wsl()
        _call_release_if_loaded("modules.best_ocr", "release_best_ocr_model")
        _call_release_if_loaded("modules.qari_ocr", "release_qari_model")
        _release_unlimited_ocr_if_loaded()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            logger.debug("[Shutdown] CUDA cache cleanup unavailable.", exc_info=True)
        logger.info("[Shutdown] GPU resources released.")
    finally:
        _cleanup_lock.release()
