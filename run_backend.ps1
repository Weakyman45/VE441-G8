param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$DbPath = Join-Path $Root "catalog.db"
$EnvPath = Join-Path $Root "backend\.env"

Write-Host "VoiceShop++ backend"
Write-Host "Project: $Root"
Write-Host "Port:    $Port"
Write-Host ""

if (-not (Test-Path $DbPath)) {
    Write-Error @"
catalog database not found:
  $DbPath

Build or copy the database first:
  Place the enriched catalog.db in the project root, or rebuild it there.
"@
}

if (-not (Test-Path $EnvPath)) {
    Write-Warning "backend\.env not found. Realtime voice and image LLM parsing need DASHSCOPE_API_KEY."
    Write-Host "Create: $EnvPath"
    Write-Host "with:   DASHSCOPE_API_KEY=your_key_here"
    Write-Host ""
}

$PythonCmd = Get-Command py -ErrorAction SilentlyContinue
if ($PythonCmd) {
    $PythonArgs = @("-3", "backend\server.py")
} else {
    $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCmd) {
        Write-Error "Python was not found. Install Python 3.10+ or add it to PATH."
    }
    $PythonArgs = @("backend\server.py")
}

Write-Host "Starting backend..."
Write-Host "URL:    http://127.0.0.1:$Port"
Write-Host "Health: http://127.0.0.1:$Port/health"
Write-Host ""

& $PythonCmd.Source @PythonArgs --host 0.0.0.0 --port $Port --db $DbPath
