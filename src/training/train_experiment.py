import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(r"C:\Users\YOUR_USER\music_source_separation")
STAGING_ROOT = Path(__file__).resolve().parents[2]
for p in (str(PROJECT_ROOT), str(STAGING_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
import src
for sub in ("data", "models", "training", "evaluation", "inference"):
    mod_name = f"src.{sub}"
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
        if hasattr(mod, "__path__") and str(STAGING_ROOT / "src" / sub) not in mod.__path__:
            mod.__path__.append(str(STAGING_ROOT / "src" / sub))
if str(STAGING_ROOT / "src") not in src.__path__:
    src.__path__.append(str(STAGING_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from config import CHECKPOINTS_DIR, VALIDATION_SAMPLES
from src.audio.stft import compute_magnitude, compute_stft
from src.data.dataset_v2 import AugmentedMUSDBDataset
from src.data.splits import create_train_validation_split
from src.models.unet_v2 import UNetV2
from src.training.losses import SeparationLoss
from src.training.losses_v2 import CompositeSeparationLoss
from src.training.experiment_selection import (
    SELECTION_METRIC,
    SelectionState,
    initialize_best_checkpoint,
    restore_selection_state,
    selection_value,
    should_stop,
    update_selection,
)


@torch.no_grad()
def evaluate_validation_set(
    model: torch.nn.Module,
    val_loader: DataLoader,
    composite_loss_fn: CompositeSeparationLoss,
    baseline_l1_fn: SeparationLoss,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute both the original baseline Linear L1 validation loss (for direct apples-to-apples
    comparison against best_model.pt's 0.273431) and the composite validation loss components.
    """
    model.eval()
    total_baseline_l1 = 0.0
    total_base_composite = 0.0
    total_log_l1 = 0.0
    total_wav_l1 = 0.0
    total_leakage = 0.0
    total_objective = 0.0

    for batch in val_loader:
        mixture = batch["mixture"].to(device, non_blocking=True)
        vocals = batch["vocals"].to(device, non_blocking=True)
        instrumental = batch["instrumental"].to(device, non_blocking=True)

        mixture_stft = compute_stft(mixture)
        vocals_stft = compute_stft(vocals)
        instrumental_stft = compute_stft(instrumental)

        mixture_mag = compute_magnitude(mixture_stft)
        vocals_mag = compute_magnitude(vocals_stft)
        instrumental_mag = compute_magnitude(instrumental_stft)

        masks = model(mixture_mag)

        base_l1 = baseline_l1_fn(
            mixture_mag, masks, vocals_mag, instrumental_mag
        )
        comp_loss, comp_metrics = composite_loss_fn(
            mixture_stft=mixture_stft,
            mixture_magnitude=mixture_mag,
            masks=masks,
            vocals_magnitude=vocals_mag,
            instrumental_magnitude=instrumental_mag,
            vocals_waveform=vocals,
            instrumental_waveform=instrumental,
        )

        total_baseline_l1 += float(base_l1.item())
        total_base_composite += comp_metrics["base_total"]
        total_log_l1 += comp_metrics["log_l1"]
        total_wav_l1 += comp_metrics.get("wav_l1", 0.0)
        total_leakage += comp_metrics["leakage"]
        total_objective += comp_metrics["objective"]

    n = max(1, len(val_loader))
    base_composite = total_base_composite / n
    objective = total_objective / n
    return {
        "val_baseline_l1": total_baseline_l1 / n,
        # Legacy alias retained for comparison with Exp04. Historical Exp04
        # selected this field, whose value was the full leakage-aware objective.
        "val_composite_loss": objective,
        "val_base_composite_loss": base_composite,
        "val_log_l1": total_log_l1 / n,
        "val_wav_l1": total_wav_l1 / n,
        "val_leakage_loss": total_leakage / n,
        "val_total_loss": objective,
    }


def run_experiment(
    exp_name: str,
    output_root: Path,
    init_weights: Optional[Path] = CHECKPOINTS_DIR / "best_model.pt",
    epochs: int = 10,
    batch_size: int = 8,
    samples_per_epoch: int = 1000,
    lr: float = 3e-4,
    min_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    base_channels: int = 16,
    pad_mode: str = "exact",
    log_input: bool = False,
    mask_mode: str = "softmax",
    augment: bool = True,
    use_bf16: bool = True,
    num_workers: int = 2,
    patience: int = 15,
    min_delta: float = 1e-4,
    max_hours: float = 12.0,
    snapshot_every: int = 10,
    min_free_disk_gb: float = 5.0,
    resume: bool = False,
    leakage_weight: float = 1.0,
    leakage_margin: float = 0.05,
    leakage_activity_threshold: float = 1e-3,
) -> Dict:
    import shutil

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_ckpt_dir = output_root / "models" / "experiments" / exp_name
    exp_log_dir = output_root / "outputs" / "logs" / exp_name

    if not resume:
        collisions = [
            directory
            for directory in (exp_ckpt_dir, exp_log_dir)
            if directory.exists() and any(directory.iterdir())
        ]
        if collisions:
            joined = ", ".join(str(path) for path in collisions)
            raise FileExistsError(
                f"Refusing to overwrite non-empty experiment output: {joined}"
            )

    exp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_log_dir.mkdir(parents=True, exist_ok=True)

    # Initial disk space safety check
    free_gb = shutil.disk_usage(output_root).free / (1024**3)
    print(f"[{exp_name}] Initial free disk space: {free_gb:.2f} GB (safety threshold: {min_free_disk_gb:.1f} GB)", flush=True)
    if free_gb < min_free_disk_gb:
        raise RuntimeError(f"Low disk space: {free_gb:.2f} GB available, required >= {min_free_disk_gb:.1f} GB")

    train_files, validation_files = create_train_validation_split()

    train_dataset = AugmentedMUSDBDataset(
        track_files=train_files,
        samples_per_epoch=samples_per_epoch,
        random_segments=True,
        augment=augment,
    )
    validation_dataset = AugmentedMUSDBDataset(
        track_files=validation_files,
        samples_per_epoch=VALIDATION_SAMPLES,
        random_segments=False,
        augment=False,
        seed=42,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = UNetV2(
        input_channels=2,
        base_channels=base_channels,
        pad_mode=pad_mode,
        log_input=log_input,
        mask_mode=mask_mode,
    ).to(device)

    composite_loss_fn = CompositeSeparationLoss()
    composite_loss_fn.leakage_weight = leakage_weight
    composite_loss_fn.leakage_margin = leakage_margin
    composite_loss_fn.leakage_activity_threshold = leakage_activity_threshold
    baseline_l1_fn = SeparationLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr
    )

    model_config = {
        "base_channels": base_channels,
        "pad_mode": pad_mode,
        "log_input": log_input,
        "mask_mode": mask_mode,
        "leakage_weight": leakage_weight,
        "leakage_margin": leakage_margin,
        "leakage_activity_threshold": leakage_activity_threshold,
    }

    start_epoch = 1
    selection_state = SelectionState(best_value=float("inf"))
    best_val_l1 = float("inf")
    history = []
    global_start_time = time.perf_counter()

    csv_path = exp_log_dir / "training_history.csv"
    config_path = exp_log_dir / "experiment_config.json"
    latest_ckpt_path = exp_ckpt_dir / "latest.pt"
    best_ckpt_path = exp_ckpt_dir / "best.pt"

    # Handle resume only from this experiment's latest.pt. Never silently
    # fall back to warm-start behavior when the caller explicitly requests resume.
    if resume:
        if not latest_ckpt_path.exists():
            raise FileNotFoundError(
                f"Cannot resume {exp_name}: checkpoint not found: {latest_ckpt_path}"
            )
        print(f"[{exp_name}] Resuming from latest checkpoint: {latest_ckpt_path}", flush=True)
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        selection_state, history = restore_selection_state(
            ckpt,
            expected_config=model_config,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_l1 = ckpt.get("best_val_l1", float("inf"))
        if "rng_state" in ckpt and isinstance(ckpt["rng_state"], torch.Tensor):
            torch.set_rng_state(ckpt["rng_state"].cpu().to(torch.uint8))
        if "cuda_rng_state" in ckpt and device.type == "cuda" and isinstance(ckpt["cuda_rng_state"], torch.Tensor):
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"].cpu().to(torch.uint8))
        if not best_ckpt_path.exists():
            raise FileNotFoundError(
                f"Cannot resume {exp_name}: preserved best checkpoint not found: {best_ckpt_path}"
            )
        print(
            f"[{exp_name}] Resumed successfully at epoch {start_epoch} "
            f"({SELECTION_METRIC}={selection_state.best_value:.6f}, "
            f"no_improvement={selection_state.epochs_without_improvement})",
            flush=True,
        )
    elif init_weights is not None and Path(init_weights).exists():
        ckpt = torch.load(init_weights, map_location=device, weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        print(f"[{exp_name}] Warm-started model weights from: {init_weights}", flush=True)

    # Initial validation evaluation
    init_val = evaluate_validation_set(
        model, validation_loader, composite_loss_fn, baseline_l1_fn, device
    )
    if not resume:
        selection_state = SelectionState(
            best_value=selection_value(init_val),
            best_epoch=0,
            epochs_without_improvement=0,
        )
        best_val_l1 = init_val["val_baseline_l1"]

        initialize_best_checkpoint(
            best_ckpt_path,
            {
                "epoch": 0,
                "exp_name": exp_name,
                "model_state_dict": model.state_dict(),
                "train_loss": None,
                "validation_loss": init_val["val_baseline_l1"],
                "val_composite_loss": init_val["val_composite_loss"],
                "train_total": None,
                "val_base_composite_loss": init_val["val_base_composite_loss"],
                "val_leakage_loss": init_val["val_leakage_loss"],
                "val_total_loss": init_val["val_total_loss"],
                "best_val_total": selection_state.best_value,
                "selection_metric": SELECTION_METRIC,
                "selection_value": selection_state.best_value,
                "selection_state": selection_state.as_dict(),
                "config": model_config,
            },
        )

    print(
        f"[{exp_name}] Initial Val | Baseline L1: {init_val['val_baseline_l1']:.6f} | "
        f"Composite: {init_val['val_composite_loss']:.6f} | LogL1: {init_val['val_log_l1']:.6f} | "
        f"WavL1: {init_val['val_wav_l1']:.6f}",
        f" | Leak: {init_val['val_leakage_loss']:.6f} | Total: {init_val['val_total_loss']:.6f}",
        flush=True,
    )

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "exp_name": exp_name,
                "init_weights": str(init_weights) if init_weights else None,
                "epochs": epochs,
                "batch_size": batch_size,
                "samples_per_epoch": samples_per_epoch,
                "lr": lr,
                "min_lr": min_lr,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
                "base_channels": base_channels,
                "pad_mode": pad_mode,
                "log_input": log_input,
                "mask_mode": mask_mode,
                "augment": augment,
                "use_bf16": use_bf16,
                "patience": patience,
                "max_hours": max_hours,
                "snapshot_every": snapshot_every,
                "leakage_weight": leakage_weight,
                "leakage_margin": leakage_margin,
                "leakage_activity_threshold": leakage_activity_threshold,
                "selection_metric": SELECTION_METRIC,
                "selection_objective": (
                    "val_base_composite_loss + leakage_weight * val_leakage_loss"
                ),
                "best_val_total": selection_state.best_value,
                "selection_state": selection_state.as_dict(),
                "initial_validation": init_val,
            },
            f,
            indent=2,
        )

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        model.train()

        running_total = 0.0
        running_lin = 0.0
        running_log = 0.0
        running_wav = 0.0
        running_leakage = 0.0
        running_objective = 0.0

        for batch_idx, batch in enumerate(train_loader):
            mixture = batch["mixture"].to(device, non_blocking=True)
            vocals = batch["vocals"].to(device, non_blocking=True)
            instrumental = batch["instrumental"].to(device, non_blocking=True)

            mixture_stft = compute_stft(mixture)
            vocals_stft = compute_stft(vocals)
            instrumental_stft = compute_stft(instrumental)

            mixture_mag = compute_magnitude(mixture_stft)
            vocals_mag = compute_magnitude(vocals_stft)
            instrumental_mag = compute_magnitude(instrumental_stft)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(use_bf16 and device.type == "cuda")
            ):
                masks = model(mixture_mag)
                loss, step_metrics = composite_loss_fn(
                    mixture_stft=mixture_stft,
                    mixture_magnitude=mixture_mag,
                    masks=masks,
                    vocals_magnitude=vocals_mag,
                    instrumental_magnitude=instrumental_mag,
                    vocals_waveform=vocals,
                    instrumental_waveform=instrumental,
                )

            # Numerical stability check
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss ({loss.item()}) at epoch {epoch}, batch {batch_idx}. Training halted."
                )

            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            running_total += step_metrics["total"]
            running_lin += step_metrics["linear_l1"]
            running_log += step_metrics["log_l1"]
            running_wav += step_metrics.get("wav_l1", 0.0)
            running_leakage += step_metrics["leakage"]
            running_objective += step_metrics["objective"]

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        n_batches = max(1, len(train_loader))
        val_stats = evaluate_validation_set(
            model, validation_loader, composite_loss_fn, baseline_l1_fn, device
        )
        elapsed = time.perf_counter() - t0
        peak_vram_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )

        row = {
            "epoch": epoch,
            "lr": round(current_lr, 7),
            "train_composite": round(running_total / n_batches, 6),
            "train_leakage": round(running_leakage / n_batches, 6),
            "train_total": round(running_objective / n_batches, 6),
            "train_linear_l1": round(running_lin / n_batches, 6),
            "train_log_l1": round(running_log / n_batches, 6),
            "train_wav_l1": round(running_wav / n_batches, 6),
            "val_baseline_l1": round(val_stats["val_baseline_l1"], 6),
            "val_composite_loss": round(val_stats["val_composite_loss"], 6),
            "val_base_composite_loss": round(val_stats["val_base_composite_loss"], 6),
            "val_leakage_loss": round(val_stats["val_leakage_loss"], 6),
            "val_total_loss": round(val_stats["val_total_loss"], 6),
            "val_log_l1": round(val_stats["val_log_l1"], 6),
            "val_wav_l1": round(val_stats["val_wav_l1"], 6),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "epoch_sec": round(elapsed, 2),
        }
        history.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)

        improved_marker = ""
        selection_state, improved = update_selection(
            selection_state,
            current_value=selection_value(val_stats),
            epoch=epoch,
            min_delta=min_delta,
        )
        if improved:
            best_val_l1 = min(best_val_l1, val_stats["val_baseline_l1"])
            improved_marker = " *BEST*"
            # Save separate best.pt
            torch.save(
                {
                    "epoch": epoch,
                    "exp_name": exp_name,
                    "model_state_dict": model.state_dict(),
                    "train_loss": row["train_composite"],
                    "validation_loss": val_stats["val_baseline_l1"],
                    "val_composite_loss": val_stats["val_composite_loss"],
                    "train_total": row["train_total"],
                    "val_base_composite_loss": val_stats["val_base_composite_loss"],
                    "val_leakage_loss": val_stats["val_leakage_loss"],
                    "val_total_loss": val_stats["val_total_loss"],
                    "best_val_total": selection_state.best_value,
                    "selection_metric": SELECTION_METRIC,
                    "selection_value": selection_state.best_value,
                    "selection_state": selection_state.as_dict(),
                    "config": model_config,
                },
                best_ckpt_path,
            )

        # Always maintain resumable latest.pt
        latest_dict = {
            "epoch": epoch,
            "exp_name": exp_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": row["train_composite"],
            "validation_loss": val_stats["val_baseline_l1"],
            "val_composite_loss": val_stats["val_composite_loss"],
            "train_total": row["train_total"],
            "val_base_composite_loss": val_stats["val_base_composite_loss"],
            "val_leakage_loss": val_stats["val_leakage_loss"],
            "val_total_loss": val_stats["val_total_loss"],
            "best_val_total": selection_state.best_value,
            # Backward-compatible alias; both fields are the full objective.
            "best_val_composite": selection_state.best_value,
            "best_val_l1": best_val_l1,
            "selection_metric": SELECTION_METRIC,
            "selection_value": selection_state.best_value,
            "selection_state": selection_state.as_dict(),
            "history": history,
            "rng_state": torch.get_rng_state(),
            "config": model_config,
        }
        if device.type == "cuda":
            latest_dict["cuda_rng_state"] = torch.cuda.get_rng_state()
        torch.save(latest_dict, latest_ckpt_path)

        # Periodic snapshot every snapshot_every epochs (avoids filling disk)
        if epoch % snapshot_every == 0:
            torch.save(
                latest_dict,
                exp_ckpt_dir / f"checkpoint_epoch_{epoch}.pt",
            )

        print(
            f"[{exp_name}] Epoch {epoch:2d}/{epochs:2d} ({elapsed:4.1f}s) | lr={current_lr:.2e} | "
            f"TrainComp={row['train_composite']:.4f} | ValComp={row['val_composite_loss']:.4f} | "
            f"ValBaseL1={row['val_baseline_l1']:.4f} | VRAM={peak_vram_mb:5.0f}MB{improved_marker}",
            f"BaseComp={row['val_base_composite_loss']:.4f} | Leak={row['val_leakage_loss']:.4f} | Total={row['val_total_loss']:.4f}",
            flush=True,
        )

        # Early stopping check
        if should_stop(selection_state, patience):
            print(
                f"[{exp_name}] Early stopping triggered: {SELECTION_METRIC} did not improve "
                f"by at least {min_delta} for {patience} epochs.",
                flush=True,
            )
            break

        # Time budget check (unattended training safety)
        elapsed_hours = (time.perf_counter() - global_start_time) / 3600.0
        if elapsed_hours >= max_hours:
            print(
                f"[{exp_name}] Time budget reached ({elapsed_hours:.2f}h >= {max_hours:.1f}h). Stopping gracefully.",
                flush=True,
            )
            break

        # Periodic disk space check
        if epoch % 5 == 0:
            free_gb = shutil.disk_usage(output_root).free / (1024**3)
            if free_gb < min_free_disk_gb:
                print(
                    f"[{exp_name}] Disk space low ({free_gb:.2f} GB < {min_free_disk_gb:.1f} GB). Stopping to prevent disk full error.",
                    flush=True,
                )
                break

    return {
        "exp_name": exp_name,
        "best_checkpoint": str(best_ckpt_path),
        "latest_checkpoint": str(latest_ckpt_path),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Modular Experiment Runner for Music Source Separation")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--init-weights", type=Path, default=CHECKPOINTS_DIR / "best_model.pt")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--pad-mode", type=str, default="exact", choices=["exact", "interpolate"])
    parser.add_argument("--log-input", action="store_true")
    parser.add_argument("--mask-mode", type=str, default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience in epochs")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Min delta for early stopping")
    parser.add_argument("--max-hours", type=float, default=12.0, help="Max budget in hours before graceful exit")
    parser.add_argument("--snapshot-every", type=int, default=10, help="Epoch interval for permanent snapshots")
    parser.add_argument("--min-free-disk-gb", type=float, default=5.0, help="Min free disk space in GB")
    parser.add_argument("--resume", action="store_true", help="Resume from latest.pt if present")
    parser.add_argument(
        "--leakage-weight",
        type=float,
        default=1.0,
        help="Multiplier for the asymmetric vocal leakage penalty",
    )
    args = parser.parse_args()

    run_experiment(
        exp_name=args.exp_name,
        output_root=args.output_root,
        init_weights=None if args.from_scratch else args.init_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        lr=args.lr,
        min_lr=args.min_lr,
        base_channels=args.base_channels,
        pad_mode=args.pad_mode,
        log_input=args.log_input,
        mask_mode=args.mask_mode,
        augment=not args.no_augment,
        use_bf16=not args.no_bf16,
        num_workers=args.num_workers,
        patience=args.patience,
        min_delta=args.min_delta,
        max_hours=args.max_hours,
        snapshot_every=args.snapshot_every,
        min_free_disk_gb=args.min_free_disk_gb,
        resume=args.resume,
        leakage_weight=args.leakage_weight,
    )

if __name__ == "__main__":
    main()
