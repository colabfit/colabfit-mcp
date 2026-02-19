import json
import subprocess
import sys
import tempfile
from pathlib import Path

from colabfit_mcp.helpers.device import detect_device


_VALID_CALCULATIONS = {"energy", "forces", "stress", "relax"}


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
        device: "cuda" or "cpu". Auto-detected if None.
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

    script = _build_run_script(
        model_path, formula, crystal_structure, lattice_constant, calcs, device
    )
    return _run_script(script)


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


def _calc_lines(calculations: list[str], mode: str) -> list[str]:
    """Return lines that run calculations and produce output."""
    lines = []
    if "relax" in calculations:
        lines += [
            "from ase.optimize import BFGS",
            "",
            f"opt = BFGS(atoms{', logfile=None' if mode == 'run' else ''})",
            "opt.run(fmax=0.01, steps=500)",
            "",
        ]
    if mode == "run":
        lines.append("_results = {}")
        if "energy" in calculations or "relax" in calculations:
            lines.append("_results['energy_eV'] = atoms.get_potential_energy()")
        if "forces" in calculations:
            lines.append("_results['forces_eV_per_Ang'] = atoms.get_forces().tolist()")
        if "stress" in calculations:
            lines.append(
                "_results['stress_GPa'] = "
                "(atoms.get_stress(voigt=True) / 160.21766).tolist()"
            )
        if "relax" in calculations:
            lines.append("_results['relaxed_positions'] = atoms.get_positions().tolist()")
            lines.append("_results['relaxed_cell'] = atoms.get_cell().tolist()")
        lines += ["", "import json", "print(json.dumps(_results))"]
    else:
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
    lines += _calc_lines(calculations, mode="snippet")
    return "\n".join(lines)


def _build_run_script(
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
    lines += _calc_lines(calculations, mode="run")
    return "\n".join(lines)


def _run_script(script: str) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp_path = Path(f.name)

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": "Calculation failed",
                "stderr": result.stderr[-2000:],
                "stdout": result.stdout[-500:],
            }

        try:
            json_line = next(
                line for line in result.stdout.splitlines() if line.startswith("{")
            )
            data = json.loads(json_line)
        except (StopIteration, json.JSONDecodeError):
            return {
                "success": False,
                "error": "Could not parse calculation output",
                "stdout": result.stdout[-2000:],
            }

        return {"success": True, "mode": "run", **data}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Calculation timed out (5 min limit)."}
    finally:
        tmp_path.unlink(missing_ok=True)
