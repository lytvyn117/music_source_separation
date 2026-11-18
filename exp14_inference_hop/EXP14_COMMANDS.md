# EXP14 commands

Run in PowerShell from `C:\Users\YOUR_USER\music_source_separation`. Use `-B` to suppress Python bytecode writes alongside imported Exp10 files.

```powershell
Set-Location -LiteralPath 'C:\Users\YOUR_USER\music_source_separation'
Get-FileHash -Algorithm SHA256 -LiteralPath exp10_scratch/runs/exp10_seed909/best_overall.pt,exp10_scratch/runs/exp10_seed909/latest.pt
Get-FileHash -Algorithm SHA256 -LiteralPath exp14_inference_hop/EXP14_DESIGN.md
nvidia-smi --query-gpu=driver_version,memory.total,memory.used,memory.free,utilization.gpu --format=csv
Get-PSDrive -PSProvider FileSystem
.\.venv\Scripts\python.exe -B .\exp14_inference_hop\evaluate_hops.py preflight
.\.venv\Scripts\python.exe -B .\exp14_inference_hop\evaluate_hops.py run 2>&1 | Tee-Object -FilePath .\exp14_inference_hop\inference.log
.\.venv\Scripts\python.exe -B .\exp14_inference_hop\report_hops.py
```

The run command executes Raw 2 s, Raw 1 s, EMA 1 s, and EMA 2 s, in that order. Both hops for each weight set use the same state dictionary from `best_overall.pt`. The runner checks for existing arm files before starting.

The first invocation of the run command stopped at the archived-row lookup during the first Raw 2 s song. It saved no arm file. Its log is `inference_attempt1.log`. The full restart uses the corrected song-row lookup and the same design and commands.
