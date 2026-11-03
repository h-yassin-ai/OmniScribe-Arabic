"""Compare Surya base with the final SFT 9750 adapter page by page."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

from modules import best_ocr
from modules.ocr_common import validate_page_range
from modules.utils import logger


ENGINE_LABELS = {
    "final": "Surya 2 · SFT 9750 final",
    "base": "Surya 2 · base",
    "all": "Comparaison OCR",
}


def get_page_outputs(pdf_path: str, page_num: int) -> tuple[str, str]:
    """Return final SFT and base Surya text for the same PDF page."""
    if not pdf_path or not page_num:
        return ("[Non transcrit]",) * 2
    page = int(page_num)
    return (
        best_ocr.get_page_text(pdf_path, page, "final"),
        best_ocr.get_page_text(pdf_path, page, "base"),
    )


def get_download_paths(pdf_path: str) -> tuple[str | None, str | None]:
    paths = (
        best_ocr.get_ocr_text_path(pdf_path, "final"),
        best_ocr.get_ocr_text_path(pdf_path, "base"),
    )
    return tuple(path if os.path.exists(path) else None for path in paths)


def _is_failure(result) -> bool:
    return isinstance(result, str) and result.lower().startswith(
        ("erreur", "échec", "echec")
    )


def run_comparison_batch(
    pdf_path: str,
    start_page: int,
    end_page: int,
    progress_cb=None,
) -> str:
    """Run the final SFT adapter and Surya base through shared vLLM."""
    if not pdf_path or not os.path.exists(pdf_path):
        return "Erreur : PDF introuvable."

    total_document_pages = best_ocr.get_pdf_total_pages(pdf_path)
    start, end = validate_page_range(start_page, end_page, total_document_pages)
    selected_pages = end - start + 1
    engines = ("final", "base")
    total_units = selected_pages * len(engines)
    comparison_started = time.perf_counter()
    completed_units = 0

    def emit(**event):
        if not progress_cb:
            return
        elapsed = time.perf_counter() - comparison_started
        event_completed = int(event.pop("comparison_completed", completed_units))
        remaining = max(0, total_units - event_completed)
        eta = (
            (elapsed / event_completed) * remaining
            if event_completed > 0
            else None
        )
        progress_cb(
            {
                **event,
                "comparison_completed": event_completed,
                "comparison_total": total_units,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "completion_time": (
                    (datetime.now() + timedelta(seconds=eta)).strftime("%H:%M:%S")
                    if eta is not None
                    else None
                ),
                **best_ocr.gpu_snapshot(),
            }
        )

    try:
        emit(
            type="comparison_start",
            stage="loading",
            engine="final",
            message=(
                f"Comparaison des pages {start}–{end} : Surya SFT 9750 final "
                "et Surya base."
            ),
        )

        for engine in engines:
            engine_offset = completed_units

            def on_surya_event(event, current_engine=engine, offset=engine_offset):
                local_completed = int(event.get("completed_pages") or 0)
                comparison_event = dict(event)
                comparison_event["engine"] = current_engine
                comparison_event["comparison_completed"] = (
                    offset + local_completed
                )
                emit(**comparison_event)

            result = best_ocr.run_ocr_batch(
                pdf_path,
                start,
                end,
                on_surya_event,
                model_variant=engine,
                parallel_pages=best_ocr.DEFAULT_PARALLEL_PAGES,
            )
            if _is_failure(result):
                raise RuntimeError(result)
            completed_units += selected_pages

        total_seconds = time.perf_counter() - comparison_started
        emit(
            type="comparison_complete",
            stage="complete",
            engine="all",
            message=(
                f"Comparaison terminée : pages {start}–{end} en "
                f"{best_ocr.format_duration(total_seconds)}."
            ),
            comparison_completed=total_units,
        )
        return f"Comparaison terminée : pages {start}–{end}."
    except Exception as exc:
        logger.exception("[OCR Compare] Comparison failed")
        emit(
            type="error",
            stage="error",
            engine="all",
            message=str(exc),
            comparison_completed=completed_units,
        )
        return f"Échec comparaison OCR : {exc}"
