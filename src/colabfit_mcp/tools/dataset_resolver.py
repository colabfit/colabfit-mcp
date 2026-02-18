from colabfit_mcp.tools.local_datasets import check_local_datasets


def resolve_train_file(
    elements: list[str] | None = None,
) -> tuple[str | None, dict]:
    """Resolve a training file from locally available datasets.

    Scans local datasets for element and property matches.
    Returns the best xyz file path if a suitable dataset is found,
    or None with a summary of what's available.

    Args:
        elements: Chemical elements to match (e.g. ["Si", "O"]).

    Returns:
        Tuple of (train_file_path, info_dict).
        On success: (path_str, dataset_info).
        On failure: (None, summary_with_guidance).
    """
    result = check_local_datasets(
        elements=elements,
        property_types=["energy", "forces"],
    )

    matches = result.get("matches", [])
    suitable = [m for m in matches if m["analysis"].get("suitable_for_training")]

    if not suitable:
        local_summary = []
        for m in result.get("matches", []):
            local_summary.append({
                "dataset_id": m["dataset_id"],
                "elements": m["analysis"].get("elements", []),
                "n_configs": m["analysis"].get("n_configs", 0),
                "has_energy": m["analysis"].get("has_energy", False),
                "has_forces": m["analysis"].get("has_forces", False),
            })

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
        local_summary = []
        for m in suitable:
            local_summary.append({
                "dataset_id": m["dataset_id"],
                "elements": m["analysis"].get("elements", []),
                "n_configs": m["analysis"].get("n_configs", 0),
            })

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

    if not best.get("xyz_files"):
        return None, {"success": False, "error": "Dataset has no xyz files."}
    train_file = best["xyz_files"][0]
    return train_file, {
        "success": True,
        "dataset_id": best["dataset_id"],
        "train_file": train_file,
        "elements": best["analysis"].get("elements", []),
        "n_configs": best["analysis"].get("n_configs", 0),
        "elements_match": elements_match,
        "auto_selected": True,
    }
