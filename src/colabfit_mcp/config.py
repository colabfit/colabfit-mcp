import os
from pathlib import Path

COLABFIT_BASE_URL = os.environ.get(
    "COLABFIT_BASE_URL", "https://materials.colabfit.org"
)
COLABFIT_AUTH = (
    os.environ.get("COLABFIT_AUTH_USER", "mcp-tool"),
    os.environ.get("COLABFIT_AUTH_PASS", "mcp-secret"),
)

DATA_ROOT = Path(os.environ.get("COLABFIT_DATA_ROOT", str(Path.home() / "colabfit")))
DOWNLOAD_DIR = DATA_ROOT / "datasets"
MODEL_DIR = DATA_ROOT / "models"

MONGODB_HOST = os.environ.get("MONGODB_HOST", "mongodb")
MONGODB_PORT = int(os.environ.get("MONGODB_PORT", "27017"))

FOUNDATION_MODEL = os.environ.get("FOUNDATION_MODEL", "small")
TORCHML_DRIVER_ID = "TorchML__MD_173118614730_001"

COLABFIT_ENERGY_KEY = "energy"
COLABFIT_FORCES_KEY = "forces"
COLABFIT_STRESS_KEY = "cauchy_stress"

FINE_TUNE_DEFAULTS = {
    "batch_size": int(os.environ.get("MACE_BATCH_SIZE", "2")),
    "lr": 0.001,
    "max_num_epochs": 50,
    "default_dtype": "float64",
    "seed": 42,
    "valid_fraction": 0.1,
    "swa": True,
    "ema": True,
    "ema_decay": 0.99,
}

TRAIN_DEFAULTS = {
    "r_max": 5.0,
    "num_channels": 128,
    "max_L": 1,
    "num_interactions": 2,
    "batch_size": int(os.environ.get("MACE_BATCH_SIZE", "10")),
    "max_num_epochs": 100,
    "default_dtype": "float64",
    "seed": 42,
    "valid_fraction": 0.1,
    "swa": True,
    "ema": True,
    "ema_decay": 0.99,
}

TRAINING_TIMEOUT = 7200
