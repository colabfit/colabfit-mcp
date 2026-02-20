import shutil

from colabfit_mcp.config import DOWNLOAD_DIR, MODEL_DIR


def check_status() -> dict:
    """Check system status including GPU, packages, and connectivity.

    Returns versions of key packages (torch, mace), GPU
    availability, disk usage, and lists of
    existing models and datasets.

    Returns:
        Dict with system info, connectivity, and resource inventory.
    """
    status = {
        "gpu": _gpu_info(),
        "packages": _package_versions(),
        "disk": _disk_info(),
        "models": _list_dir(MODEL_DIR),
        "datasets": _list_dir(DOWNLOAD_DIR),
    }
    return {"success": True, **status}


def _gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False, "note": "torch not installed"}

    from colabfit_mcp.helpers.device import detect_device

    device, name = detect_device()
    if device == "cuda":
        return {
            "available": True,
            "type": "cuda",
            "device": name,
            "memory_gb": round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            ),
        }
    if device == "mps":
        return {"available": True, "type": "mps", "device": name}
    return {"available": False}


def _package_versions() -> dict:
    versions = {}
    for pkg in ("torch", "mace"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg] = "not installed"

    return versions


def _disk_info() -> dict:
    try:
        usage = shutil.disk_usage(str(MODEL_DIR.parent))
        return {
            "total_gb": round(usage.total / 1e9, 1),
            "free_gb": round(usage.free / 1e9, 1),
        }
    except Exception:
        return {}


def _list_dir(path) -> list[str]:
    try:
        if path.exists():
            return sorted(
                p.name for p in path.iterdir() if not p.name.startswith(".")
            )
    except Exception:
        pass
    return []
