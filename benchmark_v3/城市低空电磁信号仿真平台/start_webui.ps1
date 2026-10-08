param([int]$Port = 8765, [string]$Python = "")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
$env:PYTHONIOENCODING = 'utf-8'
if (-not $Python) {
    $localPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    $Python = if (Test-Path -LiteralPath $localPython) { $localPython } else { 'python' }
}
& $Python -m webui.server --port $Port
