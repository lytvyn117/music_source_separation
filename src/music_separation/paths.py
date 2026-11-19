from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "training" / "exp10"
CACHE = PROJECT_ROOT / "data" / "development_cache"
