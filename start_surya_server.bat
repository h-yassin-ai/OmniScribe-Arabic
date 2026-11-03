@echo off
chcp 65001 >nul
title Surya OCR 2 Base + SFT 9750 Final - Serveur vLLM
cd /d "%~dp0"

echo Lancement du serveur Surya OCR 2 base + modèle final SFT 9750...
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_surya_server.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo [ERREUR] Le serveur n'a pas pu demarrer.
) else (
    echo Le serveur peut rester actif en arriere-plan.
)
pause
exit /b %EXIT_CODE%
