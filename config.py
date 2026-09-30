from pathlib import Path


# =========================================================
# PROJECT
# =========================================================

PROJECT_NAME = "music_source_separation"

PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# DATA PATHS
# =========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"

PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"


# =========================================================
# MODEL PATHS
# =========================================================

MODELS_DIR = PROJECT_ROOT / "models"

CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
FINAL_MODEL_DIR = MODELS_DIR / "final"


# =========================================================
# OUTPUT PATHS
# =========================================================

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

SEPARATED_TRACKS_DIR = OUTPUTS_DIR / "separated_tracks"
SPECTROGRAMS_DIR = OUTPUTS_DIR / "spectrograms"
LOGS_DIR = OUTPUTS_DIR / "logs"


# =========================================================
# REPORT PATHS
# =========================================================

REPORTS_DIR = PROJECT_ROOT / "reports"

FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"


# =========================================================
# AUDIO SETTINGS
# =========================================================

SAMPLE_RATE = 44100
CHANNELS = 2
SEGMENT_LENGTH_SECONDS = 6.0


# =========================================================
# STFT SETTINGS
# =========================================================

N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048


# =========================================================
# TRAINING SETTINGS
# =========================================================

BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 50
SAMPLES_PER_EPOCH = 5000


# =========================================================
# MODEL SETTINGS
# =========================================================

MODEL_TYPE = "unet"

OUTPUT_SOURCES = [
    "vocals",
    "instrumental",
]