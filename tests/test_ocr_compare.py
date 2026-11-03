from modules import ocr_compare


def test_comparison_runs_only_final_sft_and_surya_base(tmp_path, monkeypatch):
    pdf_path = tmp_path / "book.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    calls = []

    monkeypatch.setattr(
        ocr_compare.best_ocr,
        "get_pdf_total_pages",
        lambda _path: 2,
    )
    monkeypatch.setattr(
        ocr_compare.best_ocr,
        "gpu_snapshot",
        lambda: {},
    )

    def run_surya(_pdf, start, end, callback, model_variant, parallel_pages):
        calls.append(("surya", model_variant, start, end, parallel_pages))
        callback({"type": "complete", "completed_pages": end - start + 1})
        return "terminé"

    monkeypatch.setattr(ocr_compare.best_ocr, "run_ocr_batch", run_surya)

    result = ocr_compare.run_comparison_batch(str(pdf_path), 1, 2)

    assert result.startswith("Comparaison terminée")
    assert calls == [
        ("surya", "final", 1, 2, 8),
        ("surya", "base", 1, 2, 8),
    ]
