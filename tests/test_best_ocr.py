import json

from PIL import Image

from modules import best_ocr


def test_checkpoint_targets_surya_ocr_2():
    checkpoint = best_ocr.ensure_checkpoint()
    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["base_model_name_or_path"] == best_ocr.BASE_MODEL_ID
    assert config["peft_type"] == "LORA"
    assert checkpoint.name == "SFT_9750_SAFE"


def test_final_sft_inference_defaults_are_bounded_and_parallel():
    assert best_ocr.DEFAULT_MAX_NEW_TOKENS == 4096
    assert best_ocr.DEFAULT_PARALLEL_PAGES == 8


def test_base_and_final_outputs_are_kept_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(best_ocr, "OCR_RUNS_DIR", tmp_path)
    pdf_path = str(tmp_path / "book.pdf")
    assert best_ocr.get_ocr_text_path(pdf_path, "final").endswith(
        "book_surya_sft_9750.txt"
    )
    assert best_ocr.get_ocr_text_path(pdf_path, "base").endswith(
        "book_surya_base.txt"
    )
    assert (
        best_ocr.get_ocr_text_path(pdf_path, "final")
        != best_ocr.get_ocr_text_path(pdf_path, "base")
    )


def test_official_full_page_prompt_is_not_modified():
    assert best_ocr.OFFICIAL_FULL_PAGE_PROMPT == (
        "OCR this image to HTML. Each block is a div with data-label and data-bbox "
        "(x0 y0 x1 y1, normalized 0-1000)."
    )


def test_html_output_becomes_editable_plain_text():
    result = best_ocr.clean_html_ocr(
        '```html\n<div data-label="Text">مرحبا<br>بالعالم</div>\n```'
    )
    assert "مرحبا" in result
    assert "بالعالم" in result
    assert "<div" not in result


def test_surya_scaling_respects_grid_and_pixel_budget():
    image = Image.new("RGB", (3000, 4500), "white")
    scaled = best_ocr.scale_to_surya(image)
    assert scaled.width % 28 == 0
    assert scaled.height % 28 == 0
    assert scaled.width * scaled.height <= 3072 * 2048


def test_pdf_page_preserves_detail_up_to_optimized_width(monkeypatch):
    monkeypatch.setenv("BEST_OCR_MAX_IMAGE_WIDTH", "1920")
    image = Image.new("RGB", (2480, 3509), "white")
    scaled = best_ocr.prepare_surya_page(image)
    assert 1850 <= scaled.width <= 1920
    assert 2600 <= scaled.height <= 2800
    assert scaled.width % 28 == 0
    assert scaled.height % 28 == 0


def test_duration_is_human_readable():
    assert best_ocr.format_duration(3661) == "1 h 01 min 01 s"
    assert best_ocr.format_duration(None)
