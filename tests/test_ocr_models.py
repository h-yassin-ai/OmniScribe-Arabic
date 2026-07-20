import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
import torch
from PIL import Image

from modules import qari_ocr
from modules.ocr_common import (
    OCR_CONTEXT_LENGTH,
    disable_hidden_tokenizer_truncation,
    load_page_map,
    render_pdf_cache,
    save_page_map,
    validate_page_range,
)


class FakeBackendTokenizer:
    def __init__(self):
        self.truncation = {"max_length": 5120}

    def no_truncation(self):
        self.truncation = None


class FakeTokenizer:
    def __init__(self):
        self.backend_tokenizer = FakeBackendTokenizer()
        self.model_max_length = 5120
        self.init_kwargs = {}


class FakeProcessorWithTokenizer:
    def __init__(self):
        self.tokenizer = FakeTokenizer()


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeModel:
    device = torch.device("cpu")

    def generate(self, **inputs):
        prefix = inputs["input_ids"]
        suffix = torch.tensor([[7, 8]], dtype=prefix.dtype)
        return torch.cat([prefix, suffix], dim=1)


class FakeProcessor:
    def __init__(self):
        self.messages = None
        self.images = None
        self.template_kwargs = None

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        if tokenize:
            return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))
        return "formatted multimodal prompt"

    def __call__(self, *, images, **_kwargs):
        self.images = images
        return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))

    def batch_decode(self, *_args, **_kwargs):
        return ["<think>ignored</think>  بِسْمِ اللهِ  "]


class OcrModelTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.image_path = Path(self.temp_dir.name) / "page.png"
        Image.new("RGB", (80, 100), "white").save(self.image_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_qari_uses_official_punctuated_prompt(self):
        processor = FakeProcessor()
        with patch.object(qari_ocr, "_processor", processor), patch.object(qari_ocr, "_model", FakeModel()):
            result = qari_ocr._run_qari_page(str(self.image_path))
        content = processor.messages[0]["content"]
        self.assertIsInstance(content[0]["image"], Image.Image)
        self.assertEqual(content[1]["text"], "Free OCR.")
        self.assertEqual(
            processor.template_kwargs["processor_kwargs"],
            {"truncation": False, "padding": False},
        )
        self.assertEqual(result, "بِسْمِ اللهِ")

    def test_page_storage_round_trip(self):
        path = str(Path(self.temp_dir.name) / "ocr.txt")
        save_page_map(path, {2: "الثانية", 1: "الأولى"})
        self.assertEqual(load_page_map(path), {1: "الأولى", 2: "الثانية"})
        self.assertFalse(os.path.exists(path + ".tmp"))

    def test_pdf_cache_renders_at_300_dpi(self):
        pdf_path = Path(self.temp_dir.name) / "sample.pdf"
        document = fitz.open()
        document.new_page(width=72, height=72)
        document.save(pdf_path)
        document.close()

        workspace = Path(self.temp_dir.name) / "cache"
        cached_pdf, count = render_pdf_cache(str(pdf_path), str(workspace))
        with Image.open(workspace / "images" / "page_0001.png") as image:
            self.assertEqual(image.size, (300, 300))
        self.assertEqual(count, 1)
        self.assertTrue(Path(cached_pdf).exists())

    def test_page_range_validation(self):
        self.assertEqual(validate_page_range("1", 4, 5), (1, 4))
        with self.assertRaises(ValueError):
            validate_page_range(0, 2, 5)
        with self.assertRaises(ValueError):
            validate_page_range(1, 6, 5)

    def test_saved_backend_truncation_is_disabled(self):
        processor = FakeProcessorWithTokenizer()
        result = disable_hidden_tokenizer_truncation(processor)
        self.assertIs(result, processor)
        self.assertIsNone(processor.tokenizer.backend_tokenizer.truncation)
        self.assertEqual(processor.tokenizer.model_max_length, OCR_CONTEXT_LENGTH)
        self.assertEqual(
            processor.tokenizer.init_kwargs["model_max_length"],
            OCR_CONTEXT_LENGTH,
        )


if __name__ == "__main__":
    unittest.main()
