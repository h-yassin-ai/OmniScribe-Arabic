[CmdletBinding()]
param(
    [string]$Distro = "Ubuntu",
    [int]$Port = 8012,
    [double]$GpuMemoryUtilization = 0.67,
    [int]$StartupTimeoutSeconds = 420,
    [switch]$FastStartup
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$checkpoint = Join-Path $projectRoot "models\surya_sft_9750\SFT_9750_SAFE"
$outputs = Join-Path $projectRoot "outputs"
$stdoutLog = Join-Path $outputs "best_ocr_vllm.out.log"
$stderrLog = Join-Path $outputs "best_ocr_vllm.err.log"
$healthUrl = "http://127.0.0.1:$Port/health"
$apiUrl = "http://127.0.0.1:$Port/v1"
$vllmBin = "~/.venvs/al-kutub-vllm/bin/vllm"

function Test-SuryaServer {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri $healthUrl
        if ($response.StatusCode -ne 200) {
            return $false
        }
        $models = Invoke-RestMethod -TimeoutSec 3 -Uri "$apiUrl/models"
        $ids = @($models.data | ForEach-Object { $_.id })
        return ($ids -contains "surya-base") -and ($ids -contains "surya-final")
    }
    catch {
        return $false
    }
}

function Test-SuryaHealth {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri $healthUrl
        return $response.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

function Show-LogTail {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path) {
        Write-Host ""
        Write-Host "Dernières lignes de $Path" -ForegroundColor Yellow
        Get-Content -LiteralPath $Path -Tail 30
    }
}

if (Test-SuryaServer) {
    Write-Host "Surya/vLLM est déjà prêt." -ForegroundColor Green
    Write-Host "API : $apiUrl"
    exit 0
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "WSL2 est introuvable. Activez WSL avant de lancer Surya."
}

if (-not (Test-Path -LiteralPath $checkpoint)) {
    throw "Checkpoint introuvable : $checkpoint"
}

$installedDistros = (& wsl.exe --list --quiet) -replace "`0", ""
if ($installedDistros -notcontains $Distro) {
    throw "La distribution WSL '$Distro' est introuvable."
}

if (Test-SuryaHealth) {
    Write-Host "Un ancien serveur OCR est actif ; redémarrage avec le modèle final SFT 9750…" -ForegroundColor Yellow
    & wsl.exe --terminate $Distro
    Start-Sleep -Seconds 2
}

& wsl.exe -d $Distro -- bash -lc "test -x $vllmBin"
if ($LASTEXITCODE -ne 0) {
    throw "vLLM n'est pas installé dans $vllmBin."
}

$checkpointFullPath = [System.IO.Path]::GetFullPath($checkpoint)
if ($checkpointFullPath -notmatch "^([A-Za-z]):\\(.+)$") {
    throw "Chemin Windows non pris en charge : $checkpointFullPath"
}
$drive = $Matches[1].ToLowerInvariant()
$relativeCheckpoint = $Matches[2].Replace("\", "/")
$checkpointWsl = "/mnt/$drive/$relativeCheckpoint"

New-Item -ItemType Directory -Force -Path $outputs | Out-Null

$serverCommand = @(
    "exec $vllmBin serve datalab-to/surya-ocr-2"
    "--host 0.0.0.0"
    "--port $Port"
    "--dtype bfloat16"
    "--gpu-memory-utilization $GpuMemoryUtilization"
    "--max-model-len 13000"
    "--max-num-seqs 8"
    "--limit-mm-per-prompt.image 1"
    "--limit-mm-per-prompt.video 0"
    "--skip-mm-profiling"
    "--enable-lora"
    "--max-lora-rank 16"
    "--enable-tower-connector-lora"
    "--served-model-name surya-base"
    "--lora-modules surya-final='$checkpointWsl'"
) -join " "
if ($FastStartup) {
    $serverCommand += " --enforce-eager"
}

Write-Host "Démarrage de Surya OCR 2 base + modèle final SFT 9750 avec vLLM…" -ForegroundColor Cyan
if ($FastStartup) {
    Write-Host "Mode démarrage rapide : pas de capture CUDA Graphs."
}
else {
    Write-Host "Mode CUDA Graphs : démarrage plus long, puis OCR beaucoup plus rapide."
}

$process = Start-Process `
    -FilePath "wsl.exe" `
    -ArgumentList @("-d", $Distro, "--", "bash", "-lc", "`"$serverCommand`"") `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

$started = Get-Date
$lastMessageSecond = -10
while (((Get-Date) - $started).TotalSeconds -lt $StartupTimeoutSeconds) {
    if (Test-SuryaServer) {
        $elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
        Write-Host ""
        Write-Host "Surya/vLLM prêt en $elapsed s." -ForegroundColor Green
        Write-Host "API : $apiUrl"
        Write-Host "Journaux : $stdoutLog"
        exit 0
    }

    if ($process.HasExited) {
        Show-LogTail -Path $stderrLog
        Show-LogTail -Path $stdoutLog
        throw "Le serveur vLLM s'est arrêté avec le code $($process.ExitCode)."
    }

    $elapsedSeconds = [int]((Get-Date) - $started).TotalSeconds
    if ($elapsedSeconds - $lastMessageSecond -ge 10) {
        Write-Host "Initialisation GPU… $elapsedSeconds s"
        $lastMessageSecond = $elapsedSeconds
    }
    Start-Sleep -Milliseconds 500
}

Show-LogTail -Path $stderrLog
Show-LogTail -Path $stdoutLog
throw "Le serveur ne répond pas après $StartupTimeoutSeconds secondes."
