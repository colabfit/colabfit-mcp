import os
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
COLABFIT_AUTH = (_auth_user or "mcp-tool", _auth_pass or "mcp-secret")

DATA_ROOT = Path(os.environ.get("COLABFIT_DATA_ROOT", str(Path.home() / "colabfit")))
DOWNLOAD_DIR = DATA_ROOT / "datasets"
MODEL_DIR = DATA_ROOT / "models"
INFERENCE_DIR = DATA_ROOT / "inference_output"


COLABFIT_ENERGY_KEY = "energy"
COLABFIT_FORCES_KEY = "forces"
COLABFIT_STRESS_KEY = "cauchy_stress"

KLIFF_DEFAULTS = {
    "r_max": 5.0,
    "num_channels": 128,
    "lmax": 1,
    "n_layers": 2,
    "correlation": 2,
    "avg_num_neighbors": 20.0,
    "batch_size": _env_int("KLIFF_BATCH_SIZE", "4"),
    "num_workers": _env_int("KLIFF_NUM_WORKERS", "0"),
    "train_size": _env_int("TRAIN_SIZE", "0"),
    "val_size": _env_int("VAL_SIZE", "0"),
    "lr": 0.001,
    "max_num_epochs": 100,
    "seed": 42,
    "dtype": _env_str("KLIFF_DTYPE", "float32"),
}
