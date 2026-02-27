from colabfit_mcp.config import DOWNLOAD_DIR
from colabfit_mcp.tools.local_datasets import check_local_datasets


def resolve_dataset(
    elements: list[str] | None = None,
) -> tuple[dict | None, dict]:
    """Resolve a local HuggingFace-backed dataset for training.

    Scans local datasets for element and property matches.
    Returns metadata for the best matching dataset.

    Args:
        elements: Chemical elements to match (e.g. ["Si", "O"]).

    Returns:
        Tuple of (dataset_info, info_dict).
        On success: (metadata_dict, details).
            metadata_dict has: hf_id, safe_name, split, output_dir,
            dataset_id, analysis, hf_cache_dir.
        On failure: (None, summary_with_guidance).
    """
    result = check_local_datasets(
        elements=elements,
        property_types=["energy", "forces"],
    )

    matches = result.get("matches", [])
    suitable = [m for m in matches if m["analysis"].get("suitable_for_training")]

    if not suitable:
        local_summary = [
            {
                "dataset_id": m.get("dataset_id"),
                "elements": m["analysis"].get("elements", []),
                "n_configs": m["analysis"].get("n_configs", 0),
                "has_energy": m["analysis"].get("has_energy", False),
                "has_forces": m["analysis"].get("has_forces", False),
            }
            for m in result.get("matches", [])
        ]
        return None, {
            "success": False,
            "local_datasets": local_summary,
            "total_local": result.get("total_local", 0),
            "next_step": (
                "No suitable local dataset found for training. "
                "Use search_datasets to find datasets on ColabFit, "
                "then download_dataset to get training data."
            ),
        }

    best = max(suitable, key=lambda m: m["analysis"].get("n_configs", 0))

    elements_match = "exact"
    if elements:
        required = {e.capitalize() for e in elements}
        available = set(best["analysis"].get("elements", []))
        if required != available:
            elements_match = "superset" if required.issubset(available) else "partial"

    if elements_match == "partial":
        local_summary = [
            {
                "dataset_id": m.get("dataset_id"),
                "elements": m["analysis"].get("elements", []),
                "n_configs": m["analysis"].get("n_configs", 0),
            }
            for m in suitable
        ]
        return None, {
            "success": False,
            "local_datasets": local_summary,
            "elements_match": "partial",
            "next_step": (
                "Local datasets have partial element overlap with the "
                "requested elements. Consider using search_datasets to "
                "find a better match on ColabFit. If no better match "
                "exists, re-invoke with an explicit train_file path."
            ),
        }

    if not best.get("hf_id"):
        return None, {"success": False, "error": "Dataset has no HuggingFace ID."}

    dataset_info = {
        "hf_id": best["hf_id"],
        "safe_name": best.get("safe_name") or best["dataset_dir"],
        "split": best.get("split", "train"),
        "output_dir": best["output_dir"],
        "dataset_id": best["dataset_id"],
        "analysis": best["analysis"],
        "hf_cache_dir": str(DOWNLOAD_DIR / ".hf_cache"),
    }

    return dataset_info, {
        "success": True,
        "dataset_id": best["dataset_id"],
        "hf_id": best["hf_id"],
        "elements": best["analysis"].get("elements", []),
        "n_configs": best["analysis"].get("n_configs", 0),
        "elements_match": elements_match,
        "auto_selected": True,
    }
