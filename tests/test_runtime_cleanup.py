from types import SimpleNamespace

from modules import runtime_cleanup


def test_terminate_ocr_wsl_targets_only_configured_distro(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("BEST_OCR_WSL_DISTRO", "Ubuntu-Test")
    monkeypatch.setattr(runtime_cleanup.subprocess, "run", fake_run)

    assert runtime_cleanup.terminate_ocr_wsl() is True
    assert calls[0][0] == ["wsl.exe", "--terminate", "Ubuntu-Test"]
    assert calls[0][1]["timeout"] == 30
