import subprocess
from pathlib import Path

from colabfit_mcp.config import MODEL_DIR
from colabfit_mcp.helpers.kim import build_kim_model_dir, make_kim_model_name
from colabfit_mcp.helpers.xyz import extract_elements


def deploy_model(
    model_path: str,
    model_name: str = None,
    elements: list[str] = None,
    description: str = None,
) -> dict:
    """Export a MACE model to TorchScript and install as a KIM Portable Model.

    Combines TorchScript export and KIM model installation in one step.
    Uses the TorchML Model Driver for OpenKIM compatibility.

    Args:
        model_path: Path to the .model file from training.
        model_name: Human-readable model name (auto-generated if None).
        elements: Chemical elements the model supports (auto-detected
            from training data if None).
        description: Optional model description.

    Returns:
        Dict with KIM model name, install location, and usage info.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        return {
            "success": False,
            "error": f"Model file not found: {model_path}",
        }

    if model_name is None:
        model_name = model_path.stem

    if elements is None:
        elements = _try_detect_elements(model_path.parent)

    pt_path = _export_torchscript(model_path)
    if isinstance(pt_path, dict):
        return pt_path

    kim_model_name = make_kim_model_name(model_name)
    kim_dir = MODEL_DIR / "kim_models" / kim_model_name

    build_kim_model_dir(pt_path, kim_dir, kim_model_name, elements, description)

    install_result = _install_kim(kim_dir, kim_model_name, elements)
    return install_result


def _export_torchscript(model_path: Path) -> Path | dict:
    cmd = [
        "mace_create_lammps_model",
        str(model_path),
        "--dtype=float64",
        "--format=libtorch",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=model_path.parent,
            timeout=300,
        )

        default_output = model_path.parent / f"{model_path.stem}-lammps.pt"
        if default_output.exists():
            return default_output

        pt_files = list(model_path.parent.glob("*.pt"))
        if pt_files:
            return max(pt_files, key=lambda p: p.stat().st_mtime)

        return {
            "success": False,
            "error": "TorchScript export produced no .pt file",
            "stdout": (result.stdout or "")[-1000:],
            "stderr": (result.stderr or "")[-1000:],
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "mace_create_lammps_model not found. "
            "Ensure mace-torch is installed.",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "TorchScript export timed out (5 min limit).",
        }


def _install_kim(
    kim_dir: Path, kim_model_name: str, elements: list[str]
) -> dict:
    try:
        result = subprocess.run(
            [
                "kim-api-collections-management",
                "install",
                "user",
                str(kim_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            return {
                "success": True,
                "kim_model_name": kim_model_name,
                "install_dir": str(kim_dir),
                "elements": elements,
                "status": "Installed to user collection",
                "usage": f'calc = KIM("{kim_model_name}")',
                "next_step": (
                    "Model is ready. Use it with ASE via "
                    f'KIM("{kim_model_name}") or with any KIM-API '
                    "compatible simulator (LAMMPS, ASE, etc.)."
                ),
            }
        return {
            "success": False,
            "kim_model_name": kim_model_name,
            "model_dir": str(kim_dir),
            "elements": elements,
            "error": "KIM installation failed",
            "details": result.stderr,
            "next_step": (
                "Install the TorchML Model Driver from OpenKIM, then "
                f"run: kim-api-collections-management install user {kim_dir}"
            ),
        }
    except FileNotFoundError:
        return {
            "success": False,
            "kim_model_name": kim_model_name,
            "model_dir": str(kim_dir),
            "elements": elements,
            "error": "kim-api-collections-management not found",
            "next_step": (
                "Model files are prepared at the model_dir path. "
                "Install kim-api to complete KIM deployment."
            ),
        }


def _try_detect_elements(model_dir: Path) -> list[str]:
    for pattern in ("*.extxyz", "*.xyz"):
        xyz_files = list(model_dir.glob(pattern))
        if xyz_files:
            return extract_elements(xyz_files[0])
    return []
