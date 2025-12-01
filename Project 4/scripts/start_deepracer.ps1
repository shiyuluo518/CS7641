# PowerShell script to start DeepRacer container on Windows

param(
    [string]$CPUs = "3",
    [string]$Memory = "6g"
)

$ErrorActionPreference = "Stop"

Write-Host "Capping deepracer at ${CPUs} CPUs and ${Memory} memory." -ForegroundColor Cyan

# Check if Docker is available
try {
    docker --version | Out-Null
    Write-Host "Docker is available." -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker is not available. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check if Docker daemon is running
try {
    docker ps | Out-Null
    Write-Host "Docker daemon is running." -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker daemon is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

$base = "uzairakbar/deepracer:v0"
$container = "deepracer"
$image = "deepracer"
$configs = "configs"

# Function to check if image exists
function Test-DockerImage {
    param([string]$ImageName)
    $output = docker image inspect $ImageName 2>&1
    return $LASTEXITCODE -eq 0
}

# Check if base image exists
Write-Host "Checking for base image..." -ForegroundColor Yellow
if (-not (Test-DockerImage $base)) {
    Write-Host "Pulling base image $base (this may take several minutes)..." -ForegroundColor Yellow
    docker pull $base
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to pull base image." -ForegroundColor Red
        exit 1
    }
    Write-Host "Base image pulled successfully." -ForegroundColor Green
} else {
    Write-Host "Base image already exists." -ForegroundColor Green
}

# Check if deepracer image exists
Write-Host "Checking for deepracer image..." -ForegroundColor Yellow
if (-not (Test-DockerImage "${image}:latest")) {
    Write-Host "Building deepracer image (this may take a few minutes)..." -ForegroundColor Yellow
    docker build -t $image .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to build deepracer image." -ForegroundColor Red
        exit 1
    }
    docker system prune --force | Out-Null
    Write-Host "Deepracer image built successfully." -ForegroundColor Green
} else {
    Write-Host "Deepracer image already exists." -ForegroundColor Green
}

# Stop existing container if running
Write-Host "Stopping existing container (if any)..." -ForegroundColor Yellow
$ErrorActionPreference = "SilentlyContinue"
docker stop $container 2>$null | Out-Null
docker rm $container 2>$null | Out-Null
$ErrorActionPreference = "Stop"

# Start the container
Write-Host "Starting deepracer container on port 8888..." -ForegroundColor Yellow
$pwd = (Get-Location).Path
docker run --rm --detach `
    --name=$container `
    -v "${pwd}/${configs}:/${configs}:ro" `
    -p 8888:8888 `
    --cpus=$CPUs --memory=$Memory `
    $image

if ($LASTEXITCODE -eq 0) {
    Write-Host "Started deepracer Docker container successfully!" -ForegroundColor Green
    Write-Host "Container name: $container" -ForegroundColor Cyan
    Write-Host "Port: 8888" -ForegroundColor Cyan
    Write-Host "`nYou can check container status with: docker ps" -ForegroundColor Yellow
} else {
    Write-Host "ERROR: Failed to start container." -ForegroundColor Red
    exit 1
}

Start-Sleep -Seconds 2

