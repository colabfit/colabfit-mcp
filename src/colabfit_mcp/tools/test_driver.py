import importlib
import json
from pathlib import Path

from ase.io import write as ase_write

from colabfit_mcp.config import TEST_DRIVER_DIR, container_to_host
from colabfit_mcp.helpers.crystal_data import (
    _ASE_MOLECULE_NAMES,
    _CRYSTAL_STRUCTURE_EXAMPLES,
    _CRYSTAL_STRUCTURE_INFO,
)
from colabfit_mcp.helpers.kim_runner import (
    _KIMVV_TEST_DRIVERS,
    check_element_compatibility,
    load_klay_calculator,
    parse_model_params,
    run_cluster_energy_and_forces,
)
from colabfit_mcp.helpers.structures import build_atoms, validate_structure_inputs


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

    if not is_cluster:
        try:
            kimvv = importlib.import_module("kimvv")
            TestDriverClass = getattr(kimvv, test_driver_name)
        except ImportError:
            return {
                "success": False,
                "error": "kimvv is not installed. Rebuild the Docker image to include it.",
            }
        except AttributeError:
            return {
                "success": False,
                "error": f"kimvv.{test_driver_name} not found in this kimvv version.",
            }

    atoms_list = []
    results_list = []

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
        except Exception as e:
            return {"success": False, "error": f"Structure {i} ({f}): driver failed: {e}"}

        results_list.append({
            "formula": f,
            "crystal_structure": cs,
            "lattice_constant": lc,
            "properties": raw,
        })

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
