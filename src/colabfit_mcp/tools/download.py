import json
import re
import shutil
import tarfile

import requests

from colabfit_mcp.config import COLABFIT_AUTH, COLABFIT_BASE_URL, DOWNLOAD_DIR
from colabfit_mcp.helpers.xyz import analyze_xyz_files

_DATASET_ID_PATTERN = re.compile(r"^DS_[a-zA-Z0-9_]+_\d+$")


def download_dataset(dataset_id: str) -> dict:
    """Download a ColabFit dataset and analyze it for training suitability.

    Downloads the dataset as XYZ files, extracts them, determines elements,
    configuration count, whether energy/forces/stress data present.

    Args:
        dataset_id: ColabFit dataset ID (e.g. "DS_zjkz9664bapl_0").

    Returns:
        Dict with file paths, analysis results, and next_step guidance.
    """
    if not dataset_id:
        return {"success": False, "error": "dataset_id is required"}

    if not _DATASET_ID_PATTERN.match(dataset_id):
        return {
            "success": False,
            "error": f"Invalid dataset_id format: {dataset_id!r}. "
            "Expected format: DS_<alphanumeric>_<version> (e.g. DS_zjkz9664bapl_0)",
        }

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = DOWNLOAD_DIR / dataset_id
    tar_path = DOWNLOAD_DIR / f"{dataset_id}.tar.gz"

    try:
        url = f"{COLABFIT_BASE_URL}/mcp/dataset-download/xyz/" f"{dataset_id}.tar.gz"
        with requests.get(url, stream=True, auth=COLABFIT_AUTH, timeout=300) as r:
            r.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=10_000_000):
                    f.write(chunk)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(output_dir, filter="data")

        xyz_files = sorted(
            list(output_dir.rglob("*.extxyz")) + list(output_dir.rglob("*.xyz"))
        )

        metadata = None
        metadata_file = output_dir / "dataset.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

        tar_path.unlink(missing_ok=True)

        analysis = {}
        if xyz_files:
            analysis = analyze_xyz_files(xyz_files)

        next_step = _suggest_next_step(analysis)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "output_dir": str(output_dir),
            "xyz_files": [str(f) for f in xyz_files],
            "metadata": metadata,
            "analysis": analysis,
            "next_step": next_step,
        }
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"HTTP error: {e}"}
    except Exception as e:
        tar_path.unlink(missing_ok=True)
        return {"success": False, "error": str(e)}


def _suggest_next_step(analysis: dict) -> str:
    if not analysis:
        return "No XYZ files found. Check the dataset format."

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
            f"Only {n} configurations found. Fine-tuning MACE-MP-0 "
            "(fine_tune_mace) is recommended for small datasets."
        )

    return (
        "Dataset looks suitable for training. Use fine_tune_mace "
        "to fine-tune the MACE-MP-0 foundation model, or train_mace "
        "to train from scratch."
    )
