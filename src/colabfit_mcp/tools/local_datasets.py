
from colabfit_mcp.config import DOWNLOAD_DIR
from colabfit_mcp.helpers.xyz import analyze_xyz_files


def check_local_datasets(
    elements: list[str] | None = None,
    property_types: list[str] | None = None,
) -> dict:
    """Primary dataset discovery mechanism for all training workflows.

    Scans the local dataset directory for previously downloaded datasets
    and filters by element coverage and available properties. Called
    automatically by train_mace and fine_tune_mace when no train_file
    is specified. Can also be called directly to inspect local data.

    Args:
        elements: Chemical elements to filter by (e.g. ["Si", "O"]).
            Returns datasets containing ALL specified elements.
        property_types: Filter by properties like "energy",
            "atomic_forces", "cauchy_stress".

    Note:
        Only subdirectories of the download directory whose names begin with
        "DS_" (the ColabFit dataset ID prefix) are scanned.

    Returns:
        Dict with matching datasets, their analysis, and file paths.
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
        if d.is_dir() and d.name.startswith("DS_")
    ]

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
        xyz_files = sorted(
            list(ds_dir.rglob("*.extxyz")) + list(ds_dir.rglob("*.xyz"))
        )
        if not xyz_files:
            continue

        analysis = analyze_xyz_files(xyz_files)
        all_datasets.append({
            "dataset_id": ds_dir.name,
            "output_dir": str(ds_dir),
            "xyz_files": [str(f) for f in xyz_files],
            "analysis": analysis,
        })

    matches = _filter_datasets(all_datasets, elements, property_types)

    if matches:
        next_step = (
            f"Found {len(matches)} matching local dataset(s). "
            "Use fine_tune_mace or train_mace with the xyz_files paths."
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
