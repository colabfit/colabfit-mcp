import json

from colabfit_mcp.config import DOWNLOAD_DIR
from colabfit_mcp.tools.local_datasets import check_local_datasets


def resolve_dataset_by_name(name: str) -> tuple[dict | None, dict]:
    """Resolve a specific dataset by exact local folder name — no discovery, no scoring.

    Call this (via train_mace's dataset_name parameter) when you already know
    which dataset to use. It reads dataset.json directly from the named folder
    and returns exactly that dataset.

    Do NOT use resolve_dataset for this purpose. resolve_dataset scores and ranks
    all local datasets by element match and size, and may silently select a
    different dataset than the one you intend.

    Searches: DOWNLOAD_DIR/<name>/dataset.json, then DOWNLOAD_DIR/custom/<name>/dataset.json

    Return values:
        HuggingFace-backed dataset (has hf_id in metadata):
            Returns (dataset_info, {"success": True, ...})
            dataset_info has: hf_id, safe_name, split, output_dir,
            dataset_id, analysis, hf_cache_dir.

        Local extxyz dataset (no hf_id, has train_file in metadata):
            Returns (None, {"success": True, "train_file": path, ...})
            Caller (train_mace) must redirect to the extxyz loading path
            using the returned train_file value.

        Not found or misconfigured:
            Returns (None, {"success": False, "error": message})
    """
    for candidate in [DOWNLOAD_DIR / name, DOWNLOAD_DIR / "custom" / name]:
        meta_path = candidate / "dataset.json"
        if not meta_path.exists():
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        hf_id = meta.get("hf_id")
        if not hf_id:
            train_file = meta.get("train_file")
            if train_file:
                return None, {
                    "success": True,
                    "train_file": train_file,
                    "dataset_name": name,
                    "note": "Local extxyz dataset — redirecting to train_file path.",
                }
            return None, {
                "success": False,
                "error": (
                    f"Dataset '{name}' found but has no hf_id and no train_file "
                    "in its metadata. The dataset.json may be corrupt or incomplete."
                ),
            }
        return {
            "hf_id": hf_id,
            "safe_name": meta.get("safe_name") or name,
            "split": meta.get("split", "train"),
            "output_dir": str(candidate),
            "dataset_id": meta.get("dataset_id"),
            "analysis": meta.get("analysis", {}),
            "hf_cache_dir": str(DOWNLOAD_DIR / ".hf_cache"),
        }, {
            "success": True,
            "hf_id": hf_id,
            "dataset_name": name,
        }

    return None, {
        "success": False,
        "error": (
            f"Dataset '{name}' not found locally. "
            "Use download_dataset to download it first, "
            "or call check_local_datasets to see available dataset names."
        ),
    }


def resolve_dataset(
    elements: list[str] | None = None,
) -> tuple[dict | None, dict]:
    """Discover the best matching HuggingFace-backed dataset by element criteria.

    Use this ONLY for auto-discovery when you do NOT have a specific dataset
    in mind. If you already know which dataset to use, call resolve_dataset_by_name
    instead — this function ranks and scores all local datasets and may select
    a different one than intended.

    Ranking priority (highest first):
        1. Has HuggingFace ID (hf_id not None) — always ranked above extxyz datasets
        2. Element match quality: exact match > superset > partial (when elements given)
        3. Number of configurations (larger wins as tiebreaker)

    Returns:
        (dataset_info, details) on success. dataset_info has: hf_id, safe_name,
        split, output_dir, dataset_id, analysis, hf_cache_dir.
        (None, guidance_dict) on failure.
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

    def _rank(m: dict) -> tuple:
        has_hf = 1 if m.get("hf_id") else 0
        n = m["analysis"].get("n_configs", 0)
        if not elements:
            return (has_hf, 0, n)
        required = {e.capitalize() for e in elements}
        available = set(m["analysis"].get("elements", []))
        if required == available:
            match_score = 2
        elif required.issubset(available):
            match_score = 1
        else:
            match_score = 0
        return (has_hf, match_score, n)

    best = max(suitable, key=_rank)

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
                "exists, re-invoke with an explicit train_file or dataset_name."
            ),
        }

    if not best.get("hf_id"):
        train_file = best.get("train_file")
        hint = (
            f" Use train_file='{train_file}' to train on it directly."
            if train_file
            else f" Use dataset_name='{best['dataset_dir']}' to select it explicitly."
        )
        return None, {
            "success": False,
            "error": (
                f"Best matching dataset '{best['dataset_dir']}' has no HuggingFace ID "
                f"(it is a local extxyz dataset).{hint}"
            ),
        }

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
