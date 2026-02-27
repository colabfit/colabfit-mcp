import json
from pathlib import Path

from colabfit_mcp.config import DOWNLOAD_DIR
from colabfit_mcp.helpers.kliff_utils import analyze_configs

_ALLOWED_CACHE_EXTENSIONS = {".arrow", ".parquet", ".json"}

_HF_DATASET_TRANSLATION_TABLE = str.maketrans(
    {"(": "_", ")": "_", "@": "_", "/": "_", "+": "_"}
)


def _validate_hf_cache(hf_cache_dir: str, safe_name: str) -> str | None:
    """Scan the dataset's HuggingFace cache directory and return an error message
    if any file with an extension outside {.arrow, .parquet, .json} is found.
    Files without extensions (HF blob hash-files) are allowed. Returns None on success.
    """
    cache_path = Path(hf_cache_dir) / f"datasets--colabfit--{safe_name}"
    if not cache_path.exists():
        return None
    unexpected = [
        f.name
        for f in cache_path.rglob("*")
        if f.is_file() and f.suffix and f.suffix.lower() not in _ALLOWED_CACHE_EXTENSIONS
    ]
    if unexpected:
        sample = ", ".join(unexpected[:5])
        return (
            f"Unexpected file type(s) in HuggingFace cache: {sample}. "
            "Expected only .arrow/.parquet data files and .json metadata."
        )
    return None


def download_dataset(
    dataset_name: str,
    dataset_id: str | None = None,
    split: str = "train",
    n_configs: int | None = None,
) -> dict:
    """Download a ColabFit dataset from HuggingFace using KLIFF Dataset.from_huggingface.

    Uses KLIFF's Dataset.from_huggingface which calls datasets.load_dataset
    internally and builds KLIFF Configuration objects directly. No intermediate
    extxyz file is written — data lives in the HuggingFace arrow cache and in
    a metadata JSON that train_mace uses to reload without re-downloading.

    HuggingFace column defaults (from KLIFF): positions, atomic_numbers, pbc,
    cell, energy, atomic_forces. These match the standard ColabFit HF schema.

    ## IMPORTANT: n_configs does NOT reduce dataset size

    n_configs limits how many configurations KLIFF builds into Configuration
    objects during the initial load call. It does NOT download fewer data from
    HuggingFace — the full dataset parquet/arrow files are always cached locally.
    It does NOT produce a smaller dataset suitable for training; a model trained
    with n_configs=150 on a 50,000-config dataset is trained on only 150 configs,
    which will likely underfit severely.

    To find datasets that actually contain ~100–200 configurations, use the
    max_configurations parameter in search_datasets BEFORE downloading.

    dataset_name: ColabFit dataset name as returned by search_datasets.
        Special chars ((, ), @, /, +) are replaced with _ to form the HF id.
    dataset_id: Optional DS_... ID from search results. Stored in metadata for
        traceability; used for cache-hit detection.
    split: HuggingFace dataset split. Nearly all ColabFit datasets only have
        'train' — using 'test' or 'validation' will raise an error.
    n_configs: Limits how many configurations KLIFF loads into memory. Does NOT
        limit what is downloaded or cached. Use only for quick inspection of
        dataset structure, NOT for creating training subsets. To find genuinely
        small datasets, use max_configurations in search_datasets instead.

    HuggingFace arrow cache: DOWNLOAD_DIR/.hf_cache/
    Metadata JSON:           DOWNLOAD_DIR/<safe_name>/dataset.json
    """
    if not dataset_name:
        return {"success": False, "error": "dataset_name is required"}

    safe_name = dataset_name.translate(_HF_DATASET_TRANSLATION_TABLE)
    hf_id = "colabfit/" + safe_name
    output_dir = DOWNLOAD_DIR / safe_name
    meta_path = output_dir / "dataset.json"

    if meta_path.exists():
        with open(meta_path) as f:
            cached = json.load(f)
        if dataset_id is None or cached.get("dataset_id") == dataset_id:
            return {
                "success": True,
                "cached": True,
                "dataset_name": dataset_name,
                "dataset_id": cached.get("dataset_id"),
                "hf_id": hf_id,
                "output_dir": str(output_dir),
                "dataset_ref": str(meta_path),
                "analysis": cached.get("analysis", {}),
                "next_step": _suggest_next_step(cached.get("analysis", {})),
            }

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir = str(DOWNLOAD_DIR / ".hf_cache")

    try:
        from kliff.dataset import Dataset
    except ImportError as e:
        return {"success": False, "error": f"Missing dependency: {e}. Install with pip install '.[full]'."}

    try:
        dataset = Dataset.from_huggingface(
            hf_id,
            split=split,
            n_configs=n_configs,
            cache_dir=hf_cache_dir,
        )
    except Exception as e:
        return {"success": False, "error": str(e), "hf_id_tried": hf_id}

    cache_error = _validate_hf_cache(hf_cache_dir, safe_name)
    if cache_error:
        return {"success": False, "error": cache_error, "hf_id_tried": hf_id}

    analysis = analyze_configs(dataset.configs)
    metadata = {
        "hf_id": hf_id,
        "split": split,
        "n_configs_requested": n_configs,
        "dataset_name": dataset_name,
        "safe_name": safe_name,
        "dataset_id": dataset_id,
        "analysis": analysis,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "success": True,
        "cached": False,
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "hf_id": hf_id,
        "output_dir": str(output_dir),
        "dataset_ref": str(meta_path),
        "analysis": analysis,
        "next_step": _suggest_next_step(analysis),
    }


def _suggest_next_step(analysis: dict) -> str:
    if not analysis:
        return "No configurations found. Check the dataset name."
    if not analysis.get("suitable_for_training"):
        missing = []
        if not analysis.get("has_energy"):
            missing.append("energy")
        if not analysis.get("has_forces"):
            missing.append("forces")
        return (
            f"Dataset is missing {', '.join(missing)}. "
            "Search for a dataset with energy and forces data."
        )
    n = analysis.get("n_configs", 0)
    if n < 50:
        return (
            f"Only {n} configurations found. Consider searching for a larger dataset "
            "or training with reduced epochs."
        )
    return "Dataset looks suitable for training. Use train_mace to train from scratch."
