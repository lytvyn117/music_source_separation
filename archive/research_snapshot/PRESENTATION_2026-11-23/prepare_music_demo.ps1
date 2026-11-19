[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InputAudio,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$Python
)

$ErrorActionPreference = 'Stop'

function Get-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    try {
        $hasher = [Security.Cryptography.SHA256]::Create()
        try {
            return [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
        } finally {
            $hasher.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

$source = (Resolve-Path -LiteralPath $InputAudio).ProviderPath
$weights = (Resolve-Path -LiteralPath $Checkpoint).ProviderPath
$interpreter = (Resolve-Path -LiteralPath $Python).ProviderPath
$destination = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
$runner = Join-Path $PSScriptRoot 'run_demo.ps1'
$inputHash = Get-Sha256 $source

$clock = [Diagnostics.Stopwatch]::StartNew()
& $runner -InputAudio $source -OutputDir $destination -Checkpoint $weights -Python $interpreter
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$clock.Stop()

$manifest = Get-Content -Raw -LiteralPath (Join-Path $destination 'demo_manifest.json') | ConvertFrom-Json
$referenceName = 'original_input' + [IO.Path]::GetExtension($source).ToLowerInvariant()
$reference = Join-Path $destination $referenceName
if ($manifest.input_sha256 -ne $inputHash) {
    throw 'The input changed before decoding. Repeat the rehearsal with a stable input and a fresh output folder.'
}
Copy-Item -LiteralPath $source -Destination $reference
$referenceHash = Get-Sha256 $reference
if ($referenceHash -ne $inputHash -or (Get-Sha256 $source) -ne $inputHash) {
    throw 'The source and playback copy hashes differ. Repeat the rehearsal with a stable input and a fresh output folder.'
}

function Quote-PowerShellArgument([string]$Value) {
    return "'" + $Value.Replace("'", "''") + "'"
}

$command = '& ' + (Quote-PowerShellArgument $runner) +
    ' -InputAudio ' + (Quote-PowerShellArgument $source) +
    ' -OutputDir ' + (Quote-PowerShellArgument $destination) +
    ' -Checkpoint ' + (Quote-PowerShellArgument $weights) +
    ' -Python ' + (Quote-PowerShellArgument $interpreter)
[IO.File]::WriteAllText((Join-Path $destination 'execution_command.txt'), $command + [Environment]::NewLine)

$report = [ordered]@{
    playback_reference = $referenceName
    playback_reference_sha256 = $referenceHash
    elapsed_seconds = $clock.Elapsed.TotalSeconds
    inference_seconds = $manifest.inference_seconds
    peak_gpu_allocated_mib = $manifest.peak_gpu_allocated_mib
    peak_gpu_reserved_mib = $manifest.peak_gpu_reserved_mib
    mixture_max_abs = $manifest.mixture_max_abs
    mixture_tolerance = $manifest.mixture_tolerance
    details = 'demo_manifest.json'
}
[IO.File]::WriteAllText((Join-Path $destination 'rehearsal_report.json'), ($report | ConvertTo-Json))
Write-Output "Playback reference and rehearsal report saved in $destination."
