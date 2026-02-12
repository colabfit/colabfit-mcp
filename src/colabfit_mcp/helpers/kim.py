import shutil
from pathlib import Path

from colabfit_mcp.config import TORCHML_DRIVER_ID


def sanitize_model_name(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def make_kim_model_name(name: str) -> str:
    safe = sanitize_model_name(name)
    return f"MACE_{safe}__MO_000000000000_000"


def build_kim_model_dir(
    torchscript_path: Path,
    output_dir: Path,
    kim_model_name: str,
    elements: list[str],
    description: str = None,
) -> Path:
    """Create KIM model directory with all required files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(torchscript_path, output_dir / "model.pt")

    elements_str = " ".join(f'"{e}"' for e in elements)
    desc = description or f"MACE model for {', '.join(elements)}"
    kimspec = (
        "{\n"
        f'  "kim-api-id" "{kim_model_name}"\n'
        '  "item-type" "portableModel"\n'
        f'  "driver-name" "{TORCHML_DRIVER_ID}"\n'
        f"  \"species\" [{elements_str}]\n"
        f'  "title" "{kim_model_name}"\n'
        f'  "description" "{desc}"\n'
        "}\n"
    )
    (output_dir / "kimspec.edn").write_text(kimspec)

    cmake = (
        "cmake_minimum_required(VERSION 3.10)\n\n"
        "list(APPEND CMAKE_PREFIX_PATH $ENV{KIM_API_CMAKE_PREFIX_DIR})\n"
        "find_package(KIM-API-ITEMS 2.2 REQUIRED CONFIG)\n\n"
        'kim_api_items_setup_before_project(ITEM_TYPE "portableModel")\n'
        f"project({kim_model_name})\n"
        'kim_api_items_setup_after_project(ITEM_TYPE "portableModel")\n\n'
        "add_kim_api_model_library(\n"
        "  NAME            ${PROJECT_NAME}\n"
        f'  DRIVER_NAME     "{TORCHML_DRIVER_ID}"\n'
        '  PARAMETER_FILES "model.pt"\n'
        ")\n"
    )
    (output_dir / "CMakeLists.txt").write_text(cmake)

    return output_dir
