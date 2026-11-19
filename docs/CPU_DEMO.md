# Laptop demo (CPU or CUDA)

Run from the repository root. The existing laptop environment has PyTorch CPU installed. Install the shared package into that environment if needed:

```powershell
.venv/Scripts/python.exe -m pip install -e . --no-deps
```

Use a non-silent stereo WAV or FLAC at 44.1 kHz and a new output directory:

```powershell
./demos/run_demo.ps1 -Device cpu -InputAudio "C:/Audio/song.wav" -OutputDir "outputs/demo_cpu" -Checkpoint "models/final/best_overall.pt"
```

The wrapper prefers the repository's `.venv` interpreter. Override it with `-Python` if needed. `-Device auto` is the default and chooses CUDA when available, otherwise CPU. Direct CLI equivalent:

```powershell
.venv/Scripts/python.exe -m music_separation.inference.separate --device cpu --checkpoint models/final/best_overall.pt --input-audio "C:/Audio/song.wav" --output-dir outputs/demo_cpu
```

Outputs are `vocals.wav`, `instrumental.wav` and `inference_manifest.json`. Progress is printed after each window. CPU runs use FP32, four threads and attention groups of eight by default; thread count and `cpu_group_chunk` are configured in `configs/inference/exp10.json`. CPU group size does not alter model weights or context length.

A four-second window took approximately seven seconds in the initial laptop probe. With the two-second hop, a ten-second excerpt requires four windows. Longer songs take minutes. Timings vary with laptop load and power state. Prepare a short live excerpt and full-song outputs in advance.

The CPU CLI has been checked using the frozen checkpoint and synthetic audio, including padding, overlap-add and exported waveform verification. This is a functional check, not a new quality evaluation. GPU execution must still be verified on the training machine. Training and development evaluation commands retain their CUDA requirement.
