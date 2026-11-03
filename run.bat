@echo off
chcp 65001 >nul
title Al-Kutub Automator Launcher
echo Starting Al-Kutub Automator...
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv!
    echo Please make sure you are in the correct project directory.
    pause
    exit /b
)

:: The Python launcher selects the first free port and opens the right URL.
echo Launching server...
set GRADIO_INBROWSER=1
.venv\Scripts\python.exe app.py
set "APP_EXIT_CODE=%ERRORLEVEL%"

:: Exit code 20 means an Al-Kutub instance is already using the configured port.
:: Do not stop WSL in that case because the first instance may be transcribing.
if "%APP_EXIT_CODE%"=="20" (
    echo Al-Kutub est deja ouvert sur http://127.0.0.1:7860
    start "" "http://127.0.0.1:7860"
    pause
    exit /b 0
)

:: Safety net: if Python was interrupted, stop the OCR WSL distro as well.
echo Releasing GPU and stopping the OCR server...
if defined BEST_OCR_WSL_DISTRO (
    wsl.exe --terminate "%BEST_OCR_WSL_DISTRO%" >nul 2>&1
) else (
    wsl.exe --terminate Ubuntu >nul 2>&1
)

pause
exit /b %APP_EXIT_CODE%
