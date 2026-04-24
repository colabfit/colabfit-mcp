from colabfit_mcp.config import INFERENCE_DIR
from colabfit_mcp.helpers.structures import build_atoms, validate_structure_inputs


def create_structure(
    formula: str,
    crystal_structure: str | None = None,
    lattice_constant: float | None = None,
    repeat: list[int] | None = None,
) -> dict:
    """Build an ASE Atoms structure and save it as an extxyz file.

    Creates a crystal or molecular structure and writes it to the inference
    output directory. The output file can be passed to use_model as input_file
    for running calculations, or used to verify supercell size before training.

    IMPORTANT: All paths are inside the Docker container filesystem.
    Inference output is at /home/openkim/colabfit/inference_output/.

    Args:
        formula: Chemical formula (e.g. "Si", "Fe", "NaCl").
            Only letters and digits are allowed.
        crystal_structure: ASE bulk structure type ("diamond", "fcc", "bcc",
            "rocksalt", "wurtzite", "hcp", etc.) or "molecule" for gas-phase.
        lattice_constant: Optional lattice constant in Angstroms.
            If omitted, ASE uses its built-in default for the element.
        repeat: Optional supercell repetitions as [nx, ny, nz] applied to the ASE
            primitive cell. E.g. [2, 2, 2] gives 16 atoms for Si diamond (2-atom primitive).

    Returns:
        dict with output_file (container path), formula, n_atoms,
        crystal_structure, repeat, and next_step hint for use_model.
    """
    err = validate_structure_inputs(formula, crystal_structure)
    if err:
        return {"success": False, "error": err}

    try:
        atoms = build_atoms(formula, crystal_structure, lattice_constant, repeat)
    except Exception as e:
        return {"success": False, "error": f"Failed to build structure: {e}"}

    from colabfit_mcp.helpers.naming import make_timestamp, structure_file_name
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INFERENCE_DIR / structure_file_name(formula, crystal_structure, repeat, make_timestamp())

    try:
        from ase.io import write
        write(str(output_path), atoms, format="extxyz")
    except Exception as e:
        return {"success": False, "error": f"Failed to write extxyz: {e}"}

    return {
        "success": True,
        "output_file": str(output_path),
        "formula": formula,
        "n_atoms": len(atoms),
        "crystal_structure": crystal_structure,
        "repeat": repeat,
        "next_step": (
            f"Structure saved to {output_path} ({len(atoms)} atoms). "
            "Pass output_file to use_model to run calculations, "
            "or use formula+crystal_structure+repeat directly in use_model."
        ),
    }
