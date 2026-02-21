# ============================================================
# Complete run: start app + ngrok with Docker Compose
# Run this from the project root in PowerShell (Docker must be running).
# ============================================================

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Starting app and ngrok (and required: postgres, redis, minio)..." -ForegroundColor Cyan
docker compose up -d app ngrok

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker Compose failed. Is Docker Desktop running?" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Containers started. Waiting for app to be healthy..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# Show status
docker compose ps

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "  1. Open http://localhost:4040 and copy the HTTPS ngrok URL"
Write-Host "  2. Azure Portal -> your Azure Bot -> Configuration"
Write-Host "     Set Messaging endpoint to: https://YOUR-NGROK-URL/api/messages"
Write-Host "  3. In Azure Bot, open 'Test in Web Chat' and type: help"
Write-Host ""
Write-Host "API health: http://localhost:8000/health" -ForegroundColor Gray
Write-Host "API docs:   http://localhost:8000/docs" -ForegroundColor Gray
Write-Host ""
Write-Host "To stop: docker compose down" -ForegroundColor Gray
