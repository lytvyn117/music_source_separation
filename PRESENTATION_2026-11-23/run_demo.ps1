[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InputAudio,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$Checkpoint,
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Checkpoint) {
    $Checkpoint = Join-Path $repoRoot 'exp10_scratch\runs\exp10_seed909\best_overall.pt'
}
if (-not $Python) {
    $localPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    $Python = if (Test-Path -LiteralPath $localPython -PathType Leaf) { $localPython } else { 'python' }
}

& $Python -B (Join-Path $PSScriptRoot 'demo.py') --input-audio $InputAudio --output-dir $OutputDir --checkpoint $Checkpoint
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
