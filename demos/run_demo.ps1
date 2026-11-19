param([Parameter(Mandatory=$true)][string]$InputAudio, [Parameter(Mandatory=$true)][string]$OutputDir, [Parameter(Mandatory=$true)][string]$Checkpoint, [string]$Python="python")
& $Python -m music_separation.inference.separate --input-audio $InputAudio --output-dir $OutputDir --checkpoint $Checkpoint
exit $LASTEXITCODE
