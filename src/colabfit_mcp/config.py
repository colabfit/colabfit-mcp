import os
import warnings
from pathlib import Path


def _env_int(key: str, default: str) -> int:
    value = os.environ.get(key) or default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"Environment variable {key}={value!r} is not a valid integer")


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key) or default


COLABFIT_BASE_URL = os.environ.get(
    "COLABFIT_BASE_URL", "https://materials.colabfit.org"
)

_auth_user = os.environ.get("COLABFIT_AUTH_USER")
_auth_pass = os.environ.get("COLABFIT_AUTH_PASS")
if not _auth_user or not _auth_pass:
    warnings.warn(
        "COLABFIT_AUTH_USER / COLABFIT_AUTH_PASS not set; using default credentials.",
        stacklevel=2,
    )
COLABFIT_AUTH = (_auth_user or "mcp-tool", _auth_pass or "mcp-secret")

DATA_ROOT = Path(os.environ.get("COLABFIT_DATA_ROOT", str(Path.home() / "colabfit")))
DOWNLOAD_DIR = DATA_ROOT / "datasets"
MODEL_DIR = DATA_ROOT / "models"
INFERENCE_DIR = DATA_ROOT / "inference_output"


FOUNDATION_MODEL = os.environ.get("FOUNDATION_MODEL", "small")

COLABFIT_ENERGY_KEY = "REF_energy"
COLABFIT_FORCES_KEY = "REF_forces"
COLABFIT_STRESS_KEY = "cauchy_stress"

FINE_TUNE_DEFAULTS = {
    "batch_size": _env_int("MACE_BATCH_SIZE", "8"),
    "valid_batch_size": _env_int("MACE_VALID_BATCH_SIZE", "16"),
    "lr": 0.001,
    "max_num_epochs": 50,
    "default_dtype": _env_str("MACE_DTYPE", "float32"),
    "num_workers": _env_int("MACE_NUM_WORKERS", "4"),
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
    "batch_size": _env_int("MACE_BATCH_SIZE", "16"),
    "valid_batch_size": _env_int("MACE_VALID_BATCH_SIZE", "32"),
    "max_num_epochs": 100,
    "default_dtype": _env_str("MACE_DTYPE", "float32"),
    "num_workers": _env_int("MACE_NUM_WORKERS", "4"),
    "seed": 42,
    "valid_fraction": 0.1,
    "swa": True,
    "ema": True,
    "ema_decay": 0.99,
}

TRAINING_TIMEOUT = 7200
