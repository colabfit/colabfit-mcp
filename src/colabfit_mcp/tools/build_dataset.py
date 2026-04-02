"""
MCP tool for building custom filtered datasets from ColabFit VastDB.
"""

import json
import os
from pathlib import Path

import requests

from colabfit_mcp.config import COLABFIT_AUTH, COLABFIT_BASE_URL, DOWNLOAD_DIR
from colabfit_mcp.helpers.build_utils import (
    _make_dataset_name,
    _parquet_to_extxyz,
    _poll_job,
    _read_source_dataset_ids,
    _stream_and_extract,
)

BUILD_TIMEOUT = int(os.environ.get("COLABFIT_BUILD_TIMEOUT", "900"))
_CUSTOM_DIR = DOWNLOAD_DIR / "custom"


def build_dataset(
    methods_contain: list[str] | None = None,
    software_contain: list[str] | None = None,
    dataset_ids: list[str] | None = None,
    formulae: list[str] | None = None,
    properties: list[str] | None = None,
    elements: list[str] | None = None,
    num_configs: int = 1000,
    dataset_name: str | None = None,
    preview_only: bool = False,
) -> dict:
    """Build a custom filtered dataset by querying ColabFit VastDB across all datasets.

    Filters configurations by computation method, software, formula, required
    properties, and elements present. Exports to extxyz format ready for train_mace.

    Re-running with the same dataset_name returns cached results. Use a different
    dataset_name to regenerate with new filters.

    methods_contain: Substring filters on method names (case-insensitive).
        "DFT" matches "DFT-PBE", "DFT+U", etc.
    software_contain: Substring filters on software names (case-insensitive).
        "VASP" matches "VASP 6.3.2", etc.
    dataset_ids: Restrict to specific ColabFit dataset IDs (DS_... format).
        Intersected with methods/software filters if both are provided.
    formulae: Filter by reduced chemical formula (exact match). E.g. ["TiO2", "NaCl"].
    properties: Require these properties to be present. Valid values:
        energy, atomic_forces, cauchy_stress, electronic_band_gap,
        formation_energy, adsorption_energy, atomization_energy, energy_above_hull.
        Uses boolean index columns (has_forces, has_stress) for efficient filtering.
    elements: Require all listed element symbols to be present in each configuration.
        E.g. ["Fe", "O"] returns only configs containing both Fe and O.
        Uses the element_filter index column. Unknown symbols return an error.
    num_configs: Configurations to export. Range: 100–100,000. Default: 1000.
    dataset_name: Output dataset name. Auto-generated from filters if None.
    preview_only: If True, return a count estimate without generating the dataset.
    """
    if not any([methods_contain, software_contain, dataset_ids, formulae, properties, elements]):
        return {
            "success": False,
            "error": "At least one filter is required.",
            "next_step": "Provide methods_contain, software_contain, dataset_ids, formulae, properties, or elements.",
        }

    if not (100 <= num_configs <= 100_000):
        return {
            "success": False,
            "error": "num_configs must be between 100 and 100,000.",
            "next_step": "Adjust num_configs.",
        }

    if preview_only:
        return _do_count(methods_contain, software_contain, dataset_ids, formulae, properties, elements)

    if not dataset_name:
        dataset_name = _make_dataset_name(methods_contain, software_contain, formulae, elements, num_configs)

    output_dir = _CUSTOM_DIR / dataset_name
    meta_path = output_dir / "dataset.json"

    if meta_path.exists():
        return _load_cached(meta_path, dataset_name, output_dir)

    try:
        return _generate_dataset(
            methods_contain, software_contain, dataset_ids, formulae,
            properties, elements, num_configs, dataset_name, output_dir,
        )
    except Exception as e:
        return {"success": False, "error": str(e), "next_step": "Check filters and try again."}


def _do_count(methods_contain, software_contain, dataset_ids, formulae, properties, elements) -> dict:
    payload = {k: v for k, v in {
        "methods_contain": methods_contain,
        "software_contain": software_contain,
        "dataset_ids": dataset_ids,
        "formulae": formulae,
        "properties": properties,
        "elements": elements,
    }.items() if v}
    resp = requests.post(
        f"{COLABFIT_BASE_URL}/mcp/builder/count",
        json=payload,
        auth=COLABFIT_AUTH,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "success": True,
        "preview_only": True,
        "count": data.get("count"),
        "capped": data.get("capped", False),
        "next_step": "Count obtained. Call build_dataset(preview_only=False) to generate the dataset.",
    }


def _load_cached(meta_path: Path, dataset_name: str, output_dir: Path) -> dict:
    with open(meta_path) as f:
        cached = json.load(f)
    return {
        "success": True,
        "dataset_name": dataset_name,
        "output_dir": str(output_dir),
        "n_configs": cached.get("n_configs", 0),
        "train_ready": cached.get("train_ready", False),
        "train_file": cached.get("train_file"),
        "source_dataset_ids": cached.get("source_dataset_ids", []),
        "elements": cached.get("elements", []),
        "has_energy": cached.get("has_energy", False),
        "has_forces": cached.get("has_forces", False),
        "has_stress": cached.get("has_stress", False),
        "filter_summary": cached.get("filters_applied", {}),
        "next_step": (
            f"Dataset already exists at {output_dir}. "
            "Use train_mace(train_file=...) or call again with a different dataset_name to regenerate."
        ),
    }


def _generate_dataset(
    methods_contain, software_contain, dataset_ids, formulae,
    properties, elements, num_configs, dataset_name, output_dir,
) -> dict:
    payload = {k: v for k, v in {
        "methods_contain": methods_contain,
        "software_contain": software_contain,
        "dataset_ids": dataset_ids,
        "formulae": formulae,
        "properties": properties,
        "elements": elements,
    }.items() if v}
    payload["num_configs"] = num_configs

    resp = requests.post(
        f"{COLABFIT_BASE_URL}/mcp/builder/generate",
        json=payload,
        auth=COLABFIT_AUTH,
        timeout=30,
    )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    _poll_job(job_id, COLABFIT_BASE_URL, COLABFIT_AUTH, BUILD_TIMEOUT)
    output_dir.mkdir(parents=True, exist_ok=True)
    _stream_and_extract(job_id, COLABFIT_BASE_URL, COLABFIT_AUTH, output_dir)

    extxyz_path = output_dir / "configs.extxyz"
    n_configs, found_elements, has_energy, has_forces, has_stress = _parquet_to_extxyz(
        output_dir / "configurations.parquet", extxyz_path
    )
    source_ids = _read_source_dataset_ids(output_dir)

    filters_applied = {k: v for k, v in {
        "methods_contain": methods_contain,
        "software_contain": software_contain,
        "dataset_ids": dataset_ids,
        "formulae": formulae,
        "properties": properties,
        "elements": elements,
    }.items() if v}

    dataset_meta = {
        "type": "custom_build",
        "dataset_name": dataset_name,
        "n_configs": n_configs,
        "filters_applied": filters_applied,
        "source_dataset_ids": source_ids,
        "train_ready": True,
        "train_file": str(extxyz_path),
        "has_energy": has_energy,
        "has_forces": has_forces,
        "has_stress": has_stress,
        "elements": found_elements,
        "analysis": {
            "n_configs": n_configs,
            "elements": found_elements,
            "has_energy": has_energy,
            "has_forces": has_forces,
            "has_stress": has_stress,
            "suitable_for_training": has_energy and has_forces,
        },
    }
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    return {
        "success": True,
        "dataset_name": dataset_name,
        "output_dir": str(output_dir),
        "n_configs": n_configs,
        "train_ready": True,
        "train_file": str(extxyz_path),
        "source_dataset_ids": source_ids,
        "elements": found_elements,
        "has_energy": has_energy,
        "has_forces": has_forces,
        "has_stress": has_stress,
        "filter_summary": filters_applied,
        "next_step": f"Dataset ready. Call train_mace(train_file='{extxyz_path}') to begin training.",
    }
