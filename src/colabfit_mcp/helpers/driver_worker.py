import importlib
import json
import sys
from pathlib import Path

from ase.io import write as ase_write

from colabfit_mcp.config import TEST_DRIVER_DIR, container_to_host
from colabfit_mcp.helpers.crystal_data import _ASE_MOLECULE_NAMES, _CRYSTAL_STRUCTURE_INFO
from colabfit_mcp.helpers.kim_runner import (
    check_element_compatibility,
    load_klay_calculator,
    parse_model_params,
    run_cluster_energy_and_forces,
)
from colabfit_mcp.helpers.naming import extract_model_id, make_timestamp, test_driver_dir_name
from colabfit_mcp.helpers.structures import build_atoms, validate_structure_inputs


def _execute_driver(
    structures: list[dict],
    formula: str | None,
    crystal_structure: str | None,
    lattice_constant: float | None,
    input_file: str | None,
    params: dict,
    calc,
    test_driver_name: str,
    is_cluster: bool,
) -> tuple[list, list] | dict:
    """Build structures, call driver, accumulate results.

    Returns (atoms_list, results_list) on success or {"success": False, "error": ...} on failure.
    """
    if not is_cluster:
        try:
            kimvv = importlib.import_module("kimvv")
            TestDriverClass = getattr(kimvv, test_driver_name)
        except ImportError:
            return {"success": False, "error": "kimvv is not installed. Rebuild the Docker image to include it."}
        except AttributeError:
            return {"success": False, "error": f"kimvv.{test_driver_name} not found in this kimvv version."}

    atoms_list: list = []
    results_list: list = []

    for i, struct in enumerate(structures):
        f = struct.get("formula") or formula
        cs = struct.get("crystal_structure") or crystal_structure
        lc = struct.get("lattice_constant") or lattice_constant

        compat_err = check_element_compatibility(params["species"], f)
        if compat_err:
            return {"success": False, "error": f"Structure {i} ({f}): {compat_err}"}

        if is_cluster:
            if input_file is not None and len(structures) == 1:
                try:
                    from ase.io import read
                    atoms = read(input_file)
                except Exception as e:
                    return {"success": False, "error": f"Failed to read input_file: {e}"}
            else:
                err = validate_structure_inputs(f, None)
                if err:
                    return {"success": False, "error": f"Structure {i} ({f}): {err}"}
                try:
                    atoms = build_atoms(f, "molecule", lc)
                except Exception:
                    if cs is None:
                        return {
                            "success": False,
                            "error": (
                                f"Structure {i}: '{f}' is not a recognized ASE molecule name. "
                                f"Known molecule names: {_ASE_MOLECULE_NAMES}. "
                                "For crystal formulas, also pass crystal_structure."
                            ),
                        }
                    try:
                        atoms = build_atoms(f, cs, lc)
                    except Exception as e:
                        return {"success": False, "error": f"Structure {i} ({f}): failed to build: {e}"}
        else:
            if cs is None:
                return {
                    "success": False,
                    "error": (
                        f"Structure {i} ({f}): {test_driver_name} requires crystal_structure. "
                        f"Valid values: {sorted(_CRYSTAL_STRUCTURE_INFO.keys())}."
                    ),
                }
            err = validate_structure_inputs(f, cs)
            if err:
                return {"success": False, "error": f"Structure {i} ({f}): {err}"}
            try:
                atoms = build_atoms(f, cs, lc)
            except Exception as e:
                return {"success": False, "error": f"Structure {i} ({f}): failed to build: {e}"}

        atoms.info["formula"] = f
        atoms.info["crystal_structure"] = cs or ""
        if lc is not None:
            atoms.info["lattice_constant"] = lc
        atoms.info["frame_index"] = i
        atoms_list.append(atoms)

        try:
            if is_cluster:
                raw = run_cluster_energy_and_forces(calc, atoms.copy())
            else:
                driver = TestDriverClass(calc)
                driver(atoms.copy())
                raw = driver.property_instances
        except (Exception, SystemExit) as e:
            return {"success": False, "error": f"Structure {i} ({f}): driver failed: {e}"}

        results_list.append({
            "formula": f,
            "crystal_structure": cs,
            "lattice_constant": lc,
            "properties": raw,
        })

    return atoms_list, results_list


def launch_driver_background(
    model_path: str,
    test_driver_name: str,
    formula: str | None = None,
    crystal_structure: str | None = None,
    lattice_constant: float | None = None,
    device: str | None = None,
    input_file: str | None = None,
    structures: list[dict] | None = None,
) -> dict:
    """Launch test driver as a detached subprocess. Returns immediately with job status info."""
    import subprocess

    model_id = extract_model_id(Path(model_path))
    ts = make_timestamp()
    out_dir = TEST_DRIVER_DIR / test_driver_dir_name(model_id, test_driver_name, ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    job = {
        "model_path": model_path,
        "test_driver_name": test_driver_name,
        "formula": formula,
        "crystal_structure": crystal_structure,
        "lattice_constant": lattice_constant,
        "device": device,
        "input_file": input_file,
        "structures": structures,
        "timestamp": ts,
    }
    with open(out_dir / "job.json", "w") as fh:
        json.dump(job, fh, indent=2)

    status_path = out_dir / "status.json"
    with open(status_path, "w") as fh:
        json.dump({"status": "running", "pid": None, "started": ts, "error": None}, fh, indent=2)

    proc = subprocess.Popen(
        [sys.executable, "-m", "colabfit_mcp.helpers.driver_worker", str(out_dir)],
        stdout=open(out_dir / "worker.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=str(out_dir),
    )
    with open(status_path, "w") as fh:
        json.dump({"status": "running", "pid": proc.pid, "started": ts, "error": None}, fh, indent=2)

    return {
        "success": True,
        "async": True,
        "status": "running",
        "pid": proc.pid,
        "output_dir": str(out_dir),
        "output_dir_host": container_to_host(out_dir),
        "results_file": str(out_dir / "results.json"),
        "status_file": str(status_path),
        "worker_log": str(out_dir / "worker.log"),
        "next_step": "Call check_test_driver_result with output_dir when ready.",
    }


def run_driver_job(job_dir: Path) -> None:
    """Execute a driver job from a serialized job directory. Called by background subprocess."""
    status_path = job_dir / "status.json"

    def _update_status(data: dict) -> None:
        try:
            with open(status_path) as fh:
                current = json.load(fh)
        except Exception:
            current = {}
        current.update(data)
        with open(status_path, "w") as fh:
            json.dump(current, fh, indent=2)

    with open(job_dir / "job.json") as fh:
        job = json.load(fh)

    test_driver_name = job["test_driver_name"]
    model_path = job["model_path"]
    formula = job.get("formula")
    crystal_structure = job.get("crystal_structure")
    lattice_constant = job.get("lattice_constant")
    device = job.get("device")
    input_file = job.get("input_file")
    structures = job.get("structures")
    ts = job.get("timestamp") or make_timestamp()
    is_cluster = test_driver_name == "ClusterEnergyAndForces"

    if structures is None:
        structures = [{"formula": formula, "crystal_structure": crystal_structure, "lattice_constant": lattice_constant}]

    model_dir = Path(model_path)
    params = parse_model_params(model_dir)
    model_id = extract_model_id(model_dir)

    if device is None:
        try:
            from colabfit_mcp.helpers.device import detect_device
            device, _ = detect_device()
        except Exception:
            device = "cpu"

    try:
        calc = load_klay_calculator(model_dir, device)
    except Exception as e:
        _update_status({"status": "failed", "error": f"Failed to load model: {e}"})
        raise

    result = _execute_driver(
        structures, formula, crystal_structure, lattice_constant,
        input_file, params, calc, test_driver_name, is_cluster,
    )
    if isinstance(result, dict):
        _update_status({"status": "failed", "error": result.get("error")})
        raise RuntimeError(result.get("error"))

    atoms_list, results_list = result

    extxyz_path = job_dir / "structures.extxyz"
    ase_write(str(extxyz_path), atoms_list, format="extxyz")

    json_path = job_dir / "results.json"
    with open(json_path, "w") as fh:
        json.dump(
            {
                "metadata": {
                    "model_name": model_id,
                    "test_driver": test_driver_name,
                    "timestamp": ts,
                    "n_structures": len(structures),
                },
                "results": results_list,
            },
            fh,
            indent=2,
            default=str,
        )

    _update_status({"status": "completed"})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m colabfit_mcp.helpers.driver_worker <job_dir>", file=sys.stderr)
        sys.exit(1)
    _job_dir = Path(sys.argv[1])
    try:
        run_driver_job(_job_dir)
    except BaseException as _e:
        _sp = _job_dir / "status.json"
        try:
            with open(_sp) as _fh:
                _s = json.load(_fh)
        except Exception:
            _s = {}
        if _s.get("status") == "running":
            _s.update({"status": "failed", "error": f"Worker process exited unexpectedly: {_e}"})
            with open(_sp, "w") as _fh:
                json.dump(_s, _fh, indent=2)
        raise
