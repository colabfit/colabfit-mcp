import json
import os
from pathlib import Path

from ase.io import write as ase_write

from colabfit_mcp.config import TEST_DRIVER_DIR, container_to_host
from colabfit_mcp.helpers.crystal_data import (
    _ASE_MOLECULE_NAMES,
    _CRYSTAL_STRUCTURE_EXAMPLES,
    _CRYSTAL_STRUCTURE_INFO,
)
from colabfit_mcp.helpers.driver_worker import _execute_driver
from colabfit_mcp.helpers.kim_runner import (
    _KIMVV_TEST_DRIVERS,
    load_klay_calculator,
    parse_model_params,
)


def list_test_drivers(property_keyword: str | None = None) -> dict:
    """List available kimvv test drivers.

    Args:
        property_keyword: Filter by name/description (e.g. "elastic", "vacancy"). Case-insensitive.

    Returns:
        Dict with test_drivers, valid_crystal_structures, crystal_structure_info,
        crystal_structure_examples, ase_molecule_names, workflow guidance.
    """
    drivers = [
        {"name": name, **meta} for name, meta in _KIMVV_TEST_DRIVERS.items()
    ]
    if property_keyword:
        kw = property_keyword.lower()
        drivers = [
            d for d in drivers
            if kw in d["name"].lower() or kw in d["description"].lower()
            or any(kw in p for p in d["properties"])
        ]
    return {
        "success": True,
        "test_drivers": drivers,
        "total": len(drivers),
        "valid_crystal_structures": sorted(_CRYSTAL_STRUCTURE_INFO.keys()),
        "crystal_structure_info": _CRYSTAL_STRUCTURE_INFO,
        "crystal_structure_examples": _CRYSTAL_STRUCTURE_EXAMPLES,
        "ase_molecule_names": _ASE_MOLECULE_NAMES,
        "workflow": {
            "step_1_lookup": (
                "Check crystal_structure_examples for your material (exact crystal_structure "
                "and lattice_constant). If not listed, use crystal_structure_info by stoichiometry: "
                "1-element→single-element structs, 2-el 1:1→rocksalt/zincblende/wurtzite/cesiumchloride, "
                "2-el 1:2→fluorite."
            ),
            "step_2_crystal_drivers": (
                "Crystal drivers (EquilibriumCrystalStructure, ElasticConstantsCrystal, etc.): "
                "pass formula, crystal_structure, lattice_constant. lattice_constant REQUIRED for "
                "compounds. TiO2: crystal_structure='fluorite', lattice_constant=4.59."
            ),
            "step_3_cluster_driver": (
                "ClusterEnergyAndForces: (a) known ASE molecule name → formula only; "
                "(b) crystal formula → add crystal_structure + lattice_constant. "
                "Driver auto-makes cell non-periodic. Do NOT call create_structure first."
            ),
            "multi_structure": (
                "Pass structures=[{'formula':..., 'crystal_structure':..., 'lattice_constant':...}, ...] "
                "to run multiple structures. Results saved in one subdirectory."
            ),
            "critical_mistakes_to_avoid": [
                "Do NOT use 'tetragonal' for TiO2 — tetragonal is single-element only.",
                "Do NOT use mineral names (rutile, halite, sphalerite).",
                "Do NOT omit lattice_constant for compounds (rocksalt/zincblende/wurtzite/cesiumchloride/fluorite).",
                "Do NOT call create_structure before run_test_driver.",
            ],
        },
    }


def run_test_driver(
    model_path: str,
    test_driver_name: str,
    formula: str | None = None,
    crystal_structure: str | None = None,
    lattice_constant: float | None = None,
    device: str | None = None,
    input_file: str | None = None,
    structures: list[dict] | None = None,
    async_mode: bool = False,
) -> dict:
    """Run a kimvv test driver against a trained KLAY model.

    IMPORTANT: All paths are inside the Docker container filesystem.
    Provide EITHER formula (single structure) OR structures list — mutually exclusive.

    Args:
        model_path: Container path to KIM model dir (model_path_docker from train_mace).
        test_driver_name: One of the 6 built-in driver names (use list_test_drivers()).
        formula: Chemical formula for a single structure. Mutually exclusive with structures.
        crystal_structure: ASE bulk type (bcc/fcc/diamond/fluorite/rocksalt/etc.).
            TiO2 → "fluorite", NaCl → "rocksalt", Si → "diamond". Falls back for structures
            entries that omit crystal_structure.
        lattice_constant: Angstroms. Required for compound structures (rocksalt/zincblende/
            wurtzite/cesiumchloride/fluorite). Falls back for structures entries that omit it.
        device: "cuda", "cpu", or None (auto-detected).
        input_file: Container extxyz path. ClusterEnergyAndForces only; single-structure only.
        structures: Multi-structure list. Each dict: formula (required), crystal_structure,
            lattice_constant (optional — fall through to top-level params).
        async_mode: If True, launches the driver as a background subprocess and returns
            immediately with status info. Use check_test_driver_result(output_dir)
            to retrieve results when done. Recommended for slow drivers:
            CrystalStructureAndEnergyVsPressure, GroundStateCrystalStructure,
            VacancyFormationEnergyRelaxationVolumeCrystal.

    Returns:
        Dict with success, test_driver, model_name, n_structures, output_dir,
        output_dir_host, structures_file, structures_file_host, results_file,
        results_file_host, results (inline list).
    """
    if structures is not None and formula is not None:
        return {"success": False, "error": "Provide either formula or structures, not both."}
    if structures is None and formula is None and input_file is None:
        return {"success": False, "error": "Provide either formula or structures."}

    if structures is None:
        structures = [{"formula": formula, "crystal_structure": crystal_structure,
                       "lattice_constant": lattice_constant}]

    if test_driver_name not in _KIMVV_TEST_DRIVERS:
        return {
            "success": False,
            "error": (
                f"Unknown test driver {test_driver_name!r}. "
                f"Valid options: {list(_KIMVV_TEST_DRIVERS.keys())}"
            ),
        }

    if async_mode:
        from colabfit_mcp.helpers.driver_worker import launch_driver_background
        return launch_driver_background(
            model_path=model_path,
            test_driver_name=test_driver_name,
            formula=formula,
            crystal_structure=crystal_structure,
            lattice_constant=lattice_constant,
            device=device,
            input_file=input_file,
            structures=structures,
        )

    is_cluster = test_driver_name == "ClusterEnergyAndForces"

    model_dir = Path(model_path)
    if not model_dir.exists():
        return {"success": False, "error": f"Model directory not found: {model_path}"}

    params = parse_model_params(model_dir)
    if not params.get("species"):
        return {
            "success": False,
            "error": f"Could not read model elements from kliff_graph.param in {model_path}",
        }

    if device is None:
        try:
            from colabfit_mcp.helpers.device import detect_device
            device, _ = detect_device()
        except Exception:
            device = "cpu"

    try:
        calc = load_klay_calculator(model_dir, device)
    except RuntimeError as e:
        if "CUDA context" in str(e):
            return {
                "success": False,
                "cuda_context_poisoned": True,
                "error": str(e),
                "next_step": "Restart the MCP server by restarting Claude Code.",
            }
        return {"success": False, "error": f"Failed to load model: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to load model: {e}"}

    result = _execute_driver(
        structures, formula, crystal_structure, lattice_constant,
        input_file, params, calc, test_driver_name, is_cluster,
    )
    if isinstance(result, dict):
        return result
    atoms_list, results_list = result

    from colabfit_mcp.helpers.naming import make_timestamp, extract_model_id, test_driver_dir_name
    ts = make_timestamp()
    model_id = extract_model_id(model_dir)
    out_dir = TEST_DRIVER_DIR / test_driver_dir_name(model_id, test_driver_name, ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    extxyz_path = out_dir / "structures.extxyz"
    try:
        ase_write(str(extxyz_path), atoms_list, format="extxyz")
    except Exception as e:
        return {"success": False, "error": f"Failed to write structures.extxyz: {e}"}

    json_path = out_dir / "results.json"
    try:
        with open(json_path, "w") as fh:
            json.dump({"metadata": {"model_name": model_id, "test_driver": test_driver_name,
                        "timestamp": ts, "n_structures": len(structures)},
                       "results": results_list}, fh, indent=2, default=str)
    except Exception as e:
        return {"success": False, "error": f"Driver ran but failed to save results: {e}"}

    return {
        "success": True,
        "test_driver": test_driver_name,
        "model_name": model_id,
        "n_structures": len(structures),
        "output_dir": str(out_dir),
        "output_dir_host": container_to_host(out_dir),
        "structures_file": str(extxyz_path),
        "structures_file_host": container_to_host(extxyz_path),
        "results_file": str(json_path),
        "results_file_host": container_to_host(json_path),
        "results": results_list,
    }


def check_test_driver_result(output_dir: str) -> dict:
    """Check status of an async test driver job and return results when done.

    Args:
        output_dir: Container path returned by run_test_driver with async_mode=True.

    Returns:
        Dict with status ("running", "completed", or "failed") and full inline results
        when completed (same structure as sync run_test_driver success response).
    """
    out_dir = Path(output_dir)
    status_path = out_dir / "status.json"
    if not status_path.exists():
        return {"success": False, "error": "Output dir not found or job not started"}

    with open(status_path) as fh:
        status = json.load(fh)

    job_status = status.get("status")
    pid = status.get("pid")

    if job_status == "running":
        alive = False
        if pid is not None:
            try:
                os.kill(pid, 0)
                alive = True
            except (OSError, ProcessLookupError):
                pass
        if alive:
            return {"success": True, "status": "running", "pid": pid, "process_alive": True}
        # Process is dead but status.json was never updated — self-heal
        json_path = out_dir / "results.json"
        if json_path.exists():
            # Completed but status update was missed
            status["status"] = "completed"
            with open(status_path, "w") as fh:
                json.dump(status, fh, indent=2)
            job_status = "completed"
        else:
            err = "Worker process died without writing results (check worker.log for details)"
            status.update({"status": "failed", "error": err})
            with open(status_path, "w") as fh:
                json.dump(status, fh, indent=2)
            return {"success": False, "status": "failed", "error": err, "worker_log": str(out_dir / "worker.log")}

    if job_status == "failed":
        return {"success": False, "status": "failed", "error": status.get("error")}

    if job_status == "completed":
        json_path = out_dir / "results.json"
        try:
            with open(json_path) as fh:
                data = json.load(fh)
        except Exception as e:
            return {"success": False, "error": f"Failed to read results.json: {e}"}
        meta = data.get("metadata", {})
        extxyz_path = out_dir / "structures.extxyz"
        return {
            "success": True,
            "status": "completed",
            "test_driver": meta.get("test_driver"),
            "model_name": meta.get("model_name"),
            "n_structures": meta.get("n_structures"),
            "output_dir": str(out_dir),
            "output_dir_host": container_to_host(out_dir),
            "structures_file": str(extxyz_path),
            "structures_file_host": container_to_host(extxyz_path),
            "results_file": str(json_path),
            "results_file_host": container_to_host(json_path),
            "results": data.get("results", []),
        }

    return {"success": False, "error": f"Unknown status: {job_status!r}"}
