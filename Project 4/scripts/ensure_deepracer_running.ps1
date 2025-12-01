# Helper script to ensure DeepRacer container is running before training
# This prevents training failures due to missing simulation container

$container = "deepracer"

Write-Host "Checking if DeepRacer container is running..." -ForegroundColor Cyan

$running = docker ps --filter "name=$container" --format "{{.Names}}" | Select-String -Pattern $container

if ($running) {
    Write-Host "DeepRacer container is already running." -ForegroundColor Green
    docker ps --filter "name=$container" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    exit 0
} else {
    Write-Host "DeepRacer container is not running. Starting it now..." -ForegroundColor Yellow
    $scriptPath = Join-Path $PSScriptRoot "start_deepracer.ps1"
    & $scriptPath
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Waiting 15 seconds for container to initialize..." -ForegroundColor Yellow
        Start-Sleep -Seconds 15
        Write-Host "DeepRacer container is ready!" -ForegroundColor Green
        exit 0
    } else {
        Write-Host "ERROR: Failed to start DeepRacer container." -ForegroundColor Red
        exit 1
    }
}

