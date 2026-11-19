param(
    [Parameter(Mandatory=$true)][string]$InputAudio,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [ValidateSet("auto", "cpu", "cuda")][string]$Device = "auto",
    [string]$Python
)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $localPython = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv/Scripts/python.exe'
    $Python = if (Test-Path -LiteralPath $localPython) { $localPython } else { 'python' }
}
& $Python -m music_separation.inference.separate --device $Device --input-audio $InputAudio --output-dir $OutputDir --checkpoint $Checkpoint
exit $LASTEXITCODE
