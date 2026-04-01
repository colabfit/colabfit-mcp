
import json

from colabfit_mcp.config import DOWNLOAD_DIR


def check_local_datasets(
    elements: list[str] | None = None,
    property_types: list[str] | None = None,
) -> dict:
    """Primary dataset discovery mechanism for all training workflows.

    Scans the local dataset directory for previously downloaded datasets
    and filters by element coverage and available properties. Called
    automatically by train_mace when no train_file is specified. Can
    also be called directly to inspect local data.

    Args:
        elements: Chemical elements to filter by (e.g. ["Si", "O"]).
            Returns datasets containing ALL specified elements.
        property_types: Filter by properties like "energy",
            "atomic_forces", "cauchy_stress".

    Note:
        All non-hidden subdirectories of the download directory are scanned.
        The .hf_cache directory (HuggingFace parquet cache) is excluded automatically.
        Datasets are stored as HuggingFace arrow/parquet cache with a dataset.json
        metadata file. Directories without a dataset.json are skipped.

        dataset_id in each result is the DS_... id stored in dataset.json.

        train_mace calls this automatically when train_file is None — you do
        not need to call it manually before training unless you want to inspect
        what is available.

    Returns:
        Dict with matching datasets, their analysis, and metadata file paths.
    """
    if not DOWNLOAD_DIR.is_dir():
        return {
            "success": True,
            "matches": [],
            "total_local": 0,
            "next_step": "No local datasets found. Use search_datasets to find "
            "and download_dataset to download training data.",
        }

    dataset_dirs = [
        d for d in sorted(DOWNLOAD_DIR.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    ]
    custom_dir = DOWNLOAD_DIR / "custom"
    if custom_dir.is_dir():
        dataset_dirs.extend(
            d for d in sorted(custom_dir.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        )

    if not dataset_dirs:
        return {
            "success": True,
            "matches": [],
            "total_local": 0,
            "next_step": "No local datasets found. Use search_datasets to find "
            "and download_dataset to download training data.",
        }

    all_datasets = []
    for ds_dir in dataset_dirs:
        meta_path = ds_dir / "dataset.json"
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        all_datasets.append({
            "dataset_dir": ds_dir.name,
            "dataset_id": meta.get("dataset_id"),
            "output_dir": str(ds_dir),
            "hf_id": meta.get("hf_id"),
            "safe_name": meta.get("safe_name"),
            "split": meta.get("split", "train"),
            "dataset_ref": str(meta_path),
            "analysis": meta.get("analysis", {}),
            "train_file": meta.get("train_file"),
        })

    matches = _filter_datasets(all_datasets, elements, property_types)

    if matches:
        next_step = (
            f"Found {len(matches)} matching local dataset(s). "
            "Use train_mace to train a model."
        )
    else:
        next_step = (
            f"{len(all_datasets)} local dataset(s) found but none match "
            "the requested criteria. Use search_datasets to find "
            "and download_dataset to download matching data."
        )

    return {
        "success": True,
        "matches": matches,
        "total_local": len(all_datasets),
        "next_step": next_step,
    }


def _filter_datasets(
    datasets: list[dict],
    elements: list[str] | None = None,
    property_types: list[str] | None = None,
) -> list[dict]:
    """Filter dataset list by element and property criteria."""
    results = datasets

    if elements:
        required = {e.capitalize() for e in elements}
        results = [
            ds for ds in results
            if required.issubset(set(ds["analysis"].get("elements", [])))
        ]

    if property_types:
        prop_map = {
            "energy": "has_energy",
            "atomic_forces": "has_forces",
            "forces": "has_forces",
            "cauchy_stress": "has_stress",
            "stress": "has_stress",
        }
        for prop in property_types:
            key = prop_map.get(prop.lower())
            if key:
                results = [
                    ds for ds in results
                    if ds["analysis"].get(key, False)
                ]

    return results
