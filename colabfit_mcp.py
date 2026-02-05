from mcp.server.fastmcp import FastMCP
import requests
from pathlib import Path
import subprocess
import tarfile
import json
import shutil

mcp = FastMCP("colabfit-mcp")

COLABFIT_BASE_URL = "https://materials.colabfit.org"
COLABFIT_AUTH = ("mcp-tool", "mcp-secret")
DOWNLOAD_DIR = Path.home() / "Downloads" / "colabfit"
MODEL_DIR = Path.home() / "colabfit_models"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _extract_elements_from_xyz(xyz_path: Path) -> list[str]:
    elements = set()
    with open(xyz_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                elem = parts[0]
                if elem.isalpha() and len(elem) <= 2:
                    elements.add(elem)
    return sorted(elements)


@mcp.tool()
def dataset_query(
    name: str = None,
    authors: str = None,
    description: str = None,
    elements: list[str] = None,
    exact_elements: bool = False,
    doi: str = None,
    min_co: int = None,
    max_co: int = None,
    min_elements: int = None,
    max_elements: int = None,
    min_atoms: int = None,
    max_atoms: int = None,
    property_types: list[str] = None,
    license: list[str] = None,
    equilibrium: bool = None,
    given_sort_by: str = None,
    given_sort_direction: str = "descending",
    software: list[str] = None,
    methods_text_filter: list[str] = None,
    page: int = 1,
    page_size: int = 10,
):
    """
    Queries the ColabFit material and chemical database for datasets.

    Args:
        name: dataset name
        authors: dataset authors
        description: dataset description
        elements: list of chemical elements present in dataset
        exact_elements: if True, return only datasets with exactly these elements
        doi: dataset DOI
        min_co: minimum number of configurations
        max_co: maximum number of configurations
        min_elements: minimum number of distinct elements
        max_elements: maximum number of distinct elements
        min_atoms: minimum number of atoms
        max_atoms: maximum number of atoms
        property_types: property types (e.g., 'energy', 'atomic_forces', 'cauchy_stress')
        license: dataset license (e.g., 'MIT', 'CC-BY-4.0')
        equilibrium: True for equilibrium structures only
        given_sort_by: sort by 'nconfigurations', 'nelements', 'nsites', 'name', etc.
        given_sort_direction: 'ascending' or 'descending'
        software: computation software (e.g., 'VASP', 'Gaussian')
        methods_text_filter: computational method (e.g., 'DFT-PBE')
        page: page number for pagination
        page_size: results per page

    Returns:
        Dictionary with Success status, Results list, pagination info
    """
    args_dict = locals()
    args_dict.pop("page")
    args_dict.pop("page_size")
    start = (page - 1) * page_size
    end = start + page_size
    try:
        response = requests.post(
            f"{COLABFIT_BASE_URL}/mcp/dataset-query",
            json=args_dict,
            auth=COLABFIT_AUTH
        )
        response_json = response.json()
        total_pages = (len(response_json) + page_size - 1) // page_size
        return {
            "Success": True,
            "Results": response_json[start:end],
            "Result Length": len(response_json[start:end]),
            "Page": page,
            "Page Size": page_size,
            "Total Pages": total_pages,
        }
    except Exception as e:
        return {"Success": False, "Results": str(e)}


@mcp.tool()
def download_xyz(dataset_id: str) -> dict:
    """
    Downloads a ColabFit dataset as XYZ files for MACE training.

    Args:
        dataset_id: ColabFit dataset ID (e.g., 'DS_zjkz9664bapl_0')

    Returns:
        Dictionary with Success status, path to XYZ files, and dataset metadata
    """
    if not dataset_id:
        return {"Success": False, "Error": "dataset_id is required"}

    _ensure_dir(DOWNLOAD_DIR)
    output_dir = DOWNLOAD_DIR / dataset_id
    tar_path = DOWNLOAD_DIR / f"{dataset_id}.tar.gz"

    try:
        url = f"{COLABFIT_BASE_URL}/mcp/dataset-download/xyz/{dataset_id}.tar.xz"
        with requests.get(url, stream=True, auth=COLABFIT_AUTH) as r:
            r.raise_for_status()
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=10_000_000):
                    f.write(chunk)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(output_dir, filter="data")

        xyz_files = list(output_dir.rglob("*.extxyz")) + list(output_dir.rglob("*.xyz"))
        metadata_file = output_dir / "dataset.json"
        metadata = None
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

        tar_path.unlink()

        return {
            "Success": True,
            "Output Directory": str(output_dir),
            "XYZ Files": [str(f) for f in xyz_files],
            "Metadata": metadata,
        }
    except requests.exceptions.HTTPError as e:
        return {"Success": False, "Error": f"HTTP error: {e}"}
    except Exception as e:
        return {"Success": False, "Error": str(e)}


@mcp.tool()
def train_mace(
    train_file: str,
    model_name: str = "colabfit_mace",
    r_max: float = 5.0,
    num_channels: int = 128,
    max_L: int = 1,
    num_interactions: int = 2,
    max_num_epochs: int = 100,
    batch_size: int = 10,
    device: str = None,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stress_key: str = "stress",
    valid_fraction: float = 0.1,
) -> dict:
    """
    Trains a MACE model on XYZ data.

    Args:
        train_file: Path to training XYZ file (extxyz format)
        model_name: Name for the trained model
        r_max: Cutoff radius in Angstroms (default: 5.0)
        num_channels: Number of channels/hidden features (default: 128)
        max_L: Maximum L for spherical harmonics (default: 1)
        num_interactions: Number of interaction layers (default: 2)
        max_num_epochs: Maximum training epochs (default: 100)
        batch_size: Training batch size (default: 10)
        device: 'cuda' or 'cpu' (auto-detected if None)
        energy_key: Key for energy in XYZ file (default: 'energy')
        forces_key: Key for forces in XYZ file (default: 'forces')
        stress_key: Key for stress in XYZ file (default: 'stress')
        valid_fraction: Fraction of data for validation (default: 0.1)

    Returns:
        Dictionary with Success status, model path, and training info
    """
    if not train_file or not Path(train_file).exists():
        return {"Success": False, "Error": f"Training file not found: {train_file}"}

    if device is None:
        device = _detect_device()

    model_dir = _ensure_dir(MODEL_DIR / model_name)
    hidden_irreps = f"{num_channels}x0e + {num_channels}x1o"

    cmd = [
        "mace_run_train",
        f"--name={model_name}",
        f"--train_file={train_file}",
        f"--valid_fraction={valid_fraction}",
        "--model=MACE",
        f"--hidden_irreps={hidden_irreps}",
        f"--r_max={r_max}",
        f"--num_interactions={num_interactions}",
        f"--max_num_epochs={max_num_epochs}",
        f"--batch_size={batch_size}",
        f"--device={device}",
        f"--energy_key={energy_key}",
        f"--forces_key={forces_key}",
        f"--stress_key={stress_key}",
        f"--model_dir={model_dir}",
        f"--results_dir={model_dir}",
        f"--checkpoints_dir={model_dir}",
        "--default_dtype=float64",
        "--seed=42",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
        )

        model_files = list(model_dir.glob("*.model"))
        if not model_files:
            return {
                "Success": False,
                "Error": "Training completed but no model file found",
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
            }

        model_path = max(model_files, key=lambda p: p.stat().st_mtime)

        return {
            "Success": True,
            "Model Path": str(model_path),
            "Model Directory": str(model_dir),
            "Device Used": device,
            "Configuration": {
                "r_max": r_max,
                "num_channels": num_channels,
                "max_L": max_L,
                "num_interactions": num_interactions,
                "max_num_epochs": max_num_epochs,
            },
        }
    except subprocess.TimeoutExpired:
        return {"Success": False, "Error": "Training timed out (2 hour limit)"}
    except Exception as e:
        return {"Success": False, "Error": str(e)}


@mcp.tool()
def export_torchscript(
    model_path: str,
    output_path: str = None,
    dtype: str = "float64",
) -> dict:
    """
    Converts a MACE model to TorchScript format for deployment.

    Args:
        model_path: Path to the .model file from training
        output_path: Output path for TorchScript model (default: same dir as input)
        dtype: Data type ('float32' or 'float64', default: 'float64')

    Returns:
        Dictionary with Success status and TorchScript model path
    """
    model_path = Path(model_path)
    if not model_path.exists():
        return {"Success": False, "Error": f"Model file not found: {model_path}"}

    if output_path is None:
        output_path = model_path.with_suffix(".pt")
    else:
        output_path = Path(output_path)

    cmd = [
        "mace_create_lammps_model",
        str(model_path),
        f"--dtype={dtype}",
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
        if default_output.exists() and output_path != default_output:
            shutil.move(default_output, output_path)
        elif not default_output.exists():
            pt_files = list(model_path.parent.glob("*.pt"))
            if pt_files:
                newest = max(pt_files, key=lambda p: p.stat().st_mtime)
                if output_path != newest:
                    shutil.move(newest, output_path)
            else:
                return {
                    "Success": False,
                    "Error": "Export completed but no .pt file found",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

        if not output_path.exists():
            output_path = default_output

        return {
            "Success": True,
            "TorchScript Path": str(output_path),
        }
    except subprocess.TimeoutExpired:
        return {"Success": False, "Error": "Export timed out (5 minute limit)"}
    except Exception as e:
        return {"Success": False, "Error": str(e)}


@mcp.tool()
def install_kim_model(
    torchscript_path: str,
    model_name: str,
    elements: list[str],
    description: str = None,
) -> dict:
    """
    Installs a TorchScript model as a KIM Portable Model.

    NOTE: This requires the TorchML Model Driver to be installed.
    If not available, this will prepare the model directory structure
    for manual installation.

    Args:
        torchscript_path: Path to the TorchScript .pt model file
        model_name: Name for the KIM model (will be formatted as KIM ID)
        elements: List of chemical elements the model supports
        description: Optional description of the model

    Returns:
        Dictionary with Success status, model installation path, and usage info
    """
    torchscript_path = Path(torchscript_path)
    if not torchscript_path.exists():
        return {"Success": False, "Error": f"Model file not found: {torchscript_path}"}

    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in model_name)
    kim_model_name = f"MACE_{safe_name}__MO_000000000000_000"

    model_dir = _ensure_dir(MODEL_DIR / "kim_models" / kim_model_name)

    shutil.copy(torchscript_path, model_dir / "model.pt")

    elements_str = " ".join(f'"{e}"' for e in elements)
    kimspec_content = f"""{"{"}
  "kim-api-id" "{kim_model_name}"
  "item-type" "portableModel"
  "driver-name" "TorchML__MD_000000000000_000"
  "species" [{elements_str}]
  "title" "{model_name}"
  "description" "{description or f'MACE model for {", ".join(elements)}'}"
{"}"}
"""
    with open(model_dir / "kimspec.edn", "w") as f:
        f.write(kimspec_content)

    cmake_content = f"""cmake_minimum_required(VERSION 3.10)

list(APPEND CMAKE_PREFIX_PATH $ENV{{KIM_API_CMAKE_PREFIX_DIR}})
find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)

kim_api_items_setup_before_project(ITEM_TYPE "portableModel")
project({kim_model_name})
kim_api_items_setup_after_project(ITEM_TYPE "portableModel")

add_kim_api_model_library(
  NAME            ${{PROJECT_NAME}}
  DRIVER_NAME     "TorchML__MD_000000000000_000"
  PARAMETER_FILES "model.pt"
)
"""
    with open(model_dir / "CMakeLists.txt", "w") as f:
        f.write(cmake_content)

    try:
        result = subprocess.run(
            ["kim-api-collections-management", "install", "user", str(model_dir)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            return {
                "Success": True,
                "Model Name": kim_model_name,
                "Install Location": str(model_dir),
                "Elements": elements,
                "Status": "Installed to user collection",
                "Usage": f'Use with ASE: calc = KIM("{kim_model_name}")',
            }
        else:
            return {
                "Success": False,
                "Model Name": kim_model_name,
                "Model Directory": str(model_dir),
                "Elements": elements,
                "Error": "KIM installation failed - TorchML driver may not be installed",
                "Details": result.stderr,
                "Manual Install": (
                    "To install manually:\n"
                    "1. Install TorchML Model Driver from OpenKIM\n"
                    f"2. Run: kim-api-collections-management install user {model_dir}"
                ),
            }
    except FileNotFoundError:
        return {
            "Success": False,
            "Model Name": kim_model_name,
            "Model Directory": str(model_dir),
            "Elements": elements,
            "Error": "kim-api-collections-management not found",
            "Note": "Model files prepared but KIM API not available for installation",
        }
    except Exception as e:
        return {
            "Success": False,
            "Model Name": kim_model_name,
            "Model Directory": str(model_dir),
            "Elements": elements,
            "Error": str(e),
        }
