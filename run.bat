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
pause
