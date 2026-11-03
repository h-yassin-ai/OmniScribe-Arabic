import os

from modules import ocr_common


def test_save_page_map_retries_a_temporary_windows_lock(tmp_path, monkeypatch):
    destination = tmp_path / "book.txt"
    real_replace = os.replace
    attempts = []

    def flaky_replace(source, target):
        attempts.append((source, target))
        if len(attempts) < 3:
            raise PermissionError(5, "temporary lock", str(target))
        real_replace(source, target)

    monkeypatch.setattr(ocr_common.os, "replace", flaky_replace)
    monkeypatch.setenv("OCR_SAVE_RETRY_SECONDS", "2")

    result = ocr_common.save_page_map(
        str(destination),
        {2: "page two", 1: "page one"},
    )

    assert result == str(destination)
    assert len(attempts) == 3
    assert ocr_common.load_page_map(str(destination)) == {
        1: "page one",
        2: "page two",
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_save_page_map_uses_a_unique_temporary_name(tmp_path, monkeypatch):
    destination = tmp_path / "book.txt"
    observed_sources = []
    real_replace = os.replace

    def observe_replace(source, target):
        observed_sources.append(str(source))
        real_replace(source, target)

    monkeypatch.setattr(ocr_common.os, "replace", observe_replace)
    ocr_common.save_page_map(str(destination), {1: "text"})

    assert observed_sources
    assert observed_sources[0] != str(destination) + ".tmp"
    assert ".book.txt." in observed_sources[0]

