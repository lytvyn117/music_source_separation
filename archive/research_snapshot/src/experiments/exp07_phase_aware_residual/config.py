from dataclasses import dataclass
from pathlib import Path

from config import PROJECT_ROOT


EXPERIMENT = "exp07_phase_aware_residual"


@dataclass(frozen=True)
class Exp07Paths:
    root: Path = PROJECT_ROOT

    @property
    def backbone(self) -> Path:
        return self.root / "models/experiments/exp04_asymmetric_leakage/best.pt"

    @property
    def checkpoints(self) -> Path:
        return self.root / "models/experiments" / EXPERIMENT

    @property
    def logs(self) -> Path:
        return self.root / "outputs/logs" / EXPERIMENT

    @property
    def reports(self) -> Path:
        return self.root / "reports/exp07"


@dataclass(frozen=True)
class NumericalTolerances:
    waveform_nrmse_pct: float = 0.001
    waveform_max_abs: float = 1e-5
    complex_atol: float = 1e-5
    complex_rtol: float = 4 * 2**-24
    seam_window: int = 4096


@dataclass(frozen=True)
class OracleThresholds:
    tracks: int = 15
    vocal_median_gain_db: float = 1.0
    vocal_wins: int = 10
    instrumental_max_loss_db: float = 0.5
    numerical: NumericalTolerances = NumericalTolerances()


@dataclass(frozen=True)
class SuccessThresholds:
    vocal_sdr_db: float = 7.00
    instrumental_sdr_db: float = 12.65
    vocal_bleed_coeff: float = 0.2139
    vocal_wins: int = 10
    gain_regression_db: float = 0.25
    clipped_sample_increase_pp: float = 0.01
    newly_clipped_track_pct: float = 0.1
    spectral_amplification_increase_db: float = 3.0


@dataclass(frozen=True)
class Exp07Config:
    epochs: int = 30
    samples_per_epoch: int = 2000
    batch_size: int = 4
    lr: float = 1e-4
    patience: int = 8
    max_hours: float = 3.0
    snapshot_every: int = 10
    min_free_disk_gib: float = 15.0
    max_reserved_gib: float = 10.5
    seed: int = 42
