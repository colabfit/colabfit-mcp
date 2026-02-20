import re
from datetime import datetime
from pathlib import Path

from colabfit_mcp.config import INFERENCE_DIR
from colabfit_mcp.helpers.device import detect_device


_VALID_CALCULATIONS = {"energy", "forces", "stress", "relax"}

_VALID_CRYSTAL_STRUCTURES = {
    "sc", "fcc", "bcc", "tetragonal", "bct", "hcp", "rhombohedral",
    "orthorhombic", "mcl", "diamond", "zincblende", "rocksalt",
    "cesiumchloride", "fluorite", "wurtzite", "molecule",
}

_FORMULA_RE = re.compile(r"^[A-Za-z0-9]+$")


def _validate_inputs(formula: str, crystal_structure: str | None) -> str | None:
    """Return an error message if inputs are invalid, else None."""
    if not _FORMULA_RE.fullmatch(formula):
        return f"Invalid formula {formula!r}: only letters and digits are allowed."
    if crystal_structure is not None and crystal_structure.lower() not in _VALID_CRYSTAL_STRUCTURES:
        return (
            f"Invalid crystal_structure {crystal_structure!r}. "
            f"Valid options: {sorted(_VALID_CRYSTAL_STRUCTURES)}"
        )
    return None


def use_model(
    model_path: str,
    formula: str,
    crystal_structure: str | None = None,
    lattice_constant: float | None = None,
    calculations: list[str] | None = None,
    device: str | None = None,
    mode: str = "run",
) -> dict:
    """Run ASE calculations with a trained MACE model, or generate a runnable code snippet.

    Builds an ASE Atoms object from the specified structure, attaches a
    MACECalculator loaded from model_path, and either executes the requested
    calculations immediately (mode='run') or returns a copy-pasteable Python
    script (mode='snippet').

    Args:
        model_path: Path to the .model file produced by train_mace or fine_tune_mace.
        formula: Chemical formula (e.g. "Si", "Fe", "NaCl").
        crystal_structure: ASE bulk structure type ("diamond", "fcc", "bcc",
            "rocksalt", "wurtzite", "hcp", etc.) or "molecule" for gas-phase.
        lattice_constant: Optional lattice constant in Angstroms passed to bulk().
        calculations: Subset of ["energy", "forces", "stress", "relax"].
            Defaults to ["energy", "forces"].
        device: "cuda", "mps", or "cpu". Auto-detected if None.
        mode: "run" to execute and return results, "snippet" to return a
            copy-pasteable Python script.

    Returns:
        In run mode: dict with energy (eV), forces (eV/Å), stress (GPa),
            and relaxed geometry if requested.
        In snippet mode: dict with "snippet" key containing the Python script.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        return {"success": False, "error": f"Model file not found: {model_path}"}

    error = _validate_inputs(formula, crystal_structure)
    if error:
        return {"success": False, "error": error}

    if device is None:
        device, _ = detect_device()

    calcs = [c.lower() for c in (calculations or ["energy", "forces"])]
    invalid = set(calcs) - _VALID_CALCULATIONS
    if invalid:
        return {
            "success": False,
            "error": f"Unknown calculation types: {sorted(invalid)}. "
            f"Valid options: {sorted(_VALID_CALCULATIONS)}",
        }

    if mode == "snippet":
        snippet = _build_snippet(
            model_path, formula, crystal_structure, lattice_constant, calcs, device
        )
        return {
            "success": True,
            "mode": "snippet",
            "model_path": str(model_path),
            "snippet": snippet,
            "next_step": "Paste the snippet into a Python shell or Jupyter notebook.",
        }

    return _run_calculations(
        model_path, formula, crystal_structure, lattice_constant, calcs, device
    )


def _build_atoms(formula: str, crystal_structure: str | None, lattice_constant: float | None):
    """Build and return an ASE Atoms object."""
    if crystal_structure and crystal_structure.lower() != "molecule":
        from ase.build import bulk
        kwargs = {}
        if lattice_constant is not None:
            kwargs["a"] = lattice_constant
        return bulk(formula, crystal_structure, **kwargs)
    from ase.build import molecule
    return molecule(formula)


def _write_extxyz(atoms, formula: str, crystal_structure: str | None) -> Path:
    """Write atoms with calculator results to an extxyz file, return the path."""
    from ase.io import write

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    struct_tag = (crystal_structure or "molecule").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = INFERENCE_DIR / f"{formula}_{struct_tag}_{timestamp}.extxyz"
    write(str(output_path), atoms, format="extxyz")
    return output_path


def _run_calculations(
    model_path: Path,
    formula: str,
    crystal_structure: str | None,
    lattice_constant: float | None,
    calculations: list[str],
    device: str,
) -> dict:
    """Run ASE/MACE calculations in-process and return results."""
    try:
        from mace.calculators import MACECalculator
    except ImportError:
        return {"success": False, "error": "mace-torch is not installed."}

    try:
        atoms = _build_atoms(formula, crystal_structure, lattice_constant)
    except Exception as e:
        return {"success": False, "error": f"Failed to build structure: {e}"}

    try:
        calc = MACECalculator(
            model_paths=str(model_path),
            device=device,
            default_dtype="float32",
        )
        atoms.calc = calc
    except Exception as e:
        return {"success": False, "error": f"Failed to load model: {e}"}

    results = {}
    try:
        if "relax" in calculations:
            from ase.optimize import BFGS
            opt = BFGS(atoms, logfile=None)
            converged = opt.run(fmax=0.01, steps=500)
            results["relaxation_converged"] = converged

        if "energy" in calculations or "relax" in calculations:
            results["energy_eV"] = atoms.get_potential_energy()
        if "forces" in calculations:
            results["forces_eV_per_Ang"] = atoms.get_forces().tolist()
        if "stress" in calculations:
            # ASE uses compressive-positive convention; divide by eV/Å³→GPa factor
            results["stress_GPa"] = (atoms.get_stress(voigt=True) / 160.21766).tolist()
        if "relax" in calculations:
            results["relaxed_positions"] = atoms.get_positions().tolist()
            results["relaxed_cell"] = atoms.get_cell().tolist()
    except Exception as e:
        return {"success": False, "error": f"Calculation failed: {e}"}

    try:
        output_path = _write_extxyz(atoms, formula, crystal_structure)
        results["output_file"] = str(output_path)
    except Exception as e:
        results["output_file_warning"] = f"Results computed but extxyz write failed: {e}"

    return {"success": True, "mode": "run", **results}


def _structure_import(crystal_structure: str | None) -> str:
    if crystal_structure and crystal_structure.lower() != "molecule":
        return "from ase.build import bulk"
    return "from ase.build import molecule"


def _structure_creation(
    formula: str, crystal_structure: str | None, lattice_constant: float | None
) -> str:
    if crystal_structure and crystal_structure.lower() != "molecule":
        args = f"{formula!r}, {crystal_structure!r}"
        if lattice_constant is not None:
            args += f", a={lattice_constant}"
        return f"atoms = bulk({args})"
    return f"atoms = molecule({formula!r})"


def _calc_lines(calculations: list[str]) -> list[str]:
    """Return lines that run calculations and print output."""
    lines = []
    if "relax" in calculations:
        lines += [
            "from ase.optimize import BFGS",
            "",
            "opt = BFGS(atoms)",
            "opt.run(fmax=0.01, steps=500)",
            "",
        ]
    if "energy" in calculations or "relax" in calculations:
        lines.append("print(f\"Energy: {atoms.get_potential_energy():.4f} eV\")")
    if "forces" in calculations:
        lines.append("print(f\"Forces (eV/Å):\\n{atoms.get_forces()}\")")
    if "stress" in calculations:
        lines += [
            "_stress_GPa = atoms.get_stress(voigt=True) / 160.21766",
            "print(f\"Stress (GPa): {_stress_GPa}\")",
        ]
    if "relax" in calculations:
        lines.append("print(f\"Relaxed cell (Å):\\n{atoms.cell[:]}\")")
    return lines


def _build_snippet(
    model_path: Path,
    formula: str,
    crystal_structure: str | None,
    lattice_constant: float | None,
    calculations: list[str],
    device: str,
) -> str:
    lines = [
        _structure_import(crystal_structure),
        "from mace.calculators import MACECalculator",
        "",
        _structure_creation(formula, crystal_structure, lattice_constant),
        "",
        "calc = MACECalculator(",
        f"    model_paths={str(model_path)!r},",
        f"    device={device!r},",
        "    default_dtype='float32',",
        ")",
        "atoms.calc = calc",
        "",
    ]
    lines += _calc_lines(calculations)
    return "\n".join(lines)
