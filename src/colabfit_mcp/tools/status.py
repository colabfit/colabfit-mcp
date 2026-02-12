import shutil

from colabfit_mcp.config import DOWNLOAD_DIR, MODEL_DIR, MONGODB_HOST, MONGODB_PORT


def check_status() -> dict:
    """Check system status including GPU, packages, and connectivity.

    Returns versions of key packages (torch, mace, kim-api), GPU
    availability, MongoDB connectivity, disk usage, and lists of
    existing models and datasets.

    Returns:
        Dict with system info, connectivity, and resource inventory.
    """
    status = {
        "gpu": _gpu_info(),
        "packages": _package_versions(),
        "mongodb": _check_mongodb(),
        "disk": _disk_info(),
        "models": _list_dir(MODEL_DIR),
        "datasets": _list_dir(DOWNLOAD_DIR),
    }
    return {"success": True, **status}


def _gpu_info() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "memory_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1e9, 1
                ),
            }
        return {"available": False}
    except ImportError:
        return {"available": False, "note": "torch not installed"}


def _package_versions() -> dict:
    versions = {}
    for pkg in ("torch", "mace", "kimpy", "kliff"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[pkg] = "not installed"

    try:
        import subprocess
        result = subprocess.run(
            ["kim-api-collections-management", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        versions["kim-api"] = result.stdout.strip() or "installed"
    except Exception:
        versions["kim-api"] = "not installed"

    return versions


def _check_mongodb() -> dict:
    try:
        from pymongo import MongoClient
        client = MongoClient(
            MONGODB_HOST, MONGODB_PORT, serverSelectionTimeoutMS=3000
        )
        client.admin.command("ping")
        client.close()
        return {"connected": True, "host": MONGODB_HOST, "port": MONGODB_PORT}
    except ImportError:
        return {"connected": False, "note": "pymongo not installed"}
    except Exception as e:
        return {"connected": False, "error": str(e)}


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
