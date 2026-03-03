from datetime import datetime
from pathlib import Path

from colabfit_mcp.config import INFERENCE_DIR
from colabfit_mcp.helpers.device import detect_device
from colabfit_mcp.helpers.structures import build_atoms, validate_structure_inputs

_VALID_CALCULATIONS = {"energy", "forces", "stress", "relax"}


def use_model(
    model_path: str,
    formula: str | None = None,
    crystal_structure: str | None = None,
    lattice_constant: float | None = None,
    repeat: list[int] | None = None,
    structures: list[dict] | None = None,
    input_file: str | None = None,
    calculations: list[str] | None = None,
    device: str | None = None,
    mode: str = "run",
) -> dict:
    """Run ASE calculations with a trained KLAY/KLIFF KIM model, or generate a code snippet.

    Builds one or more ASE Atoms objects, loads the KLAY model from the KIM model
    directory (produced by train_mace), and either executes the requested calculations
    immediately (mode='run') or returns a copy-pasteable Python script (mode='snippet').

    IMPORTANT: model_path must be the KIM model *subdirectory* returned as
    'model_path_docker' by train_mace (format: Name__MO_000000000000_000/), not the
    parent model_dir. The subdirectory must contain model.pt and kliff_graph.param.

    IMPORTANT: All paths are inside the Docker container filesystem.

    Exactly one structure source must be provided:
        - formula + crystal_structure [+ repeat]: single or supercell structure
        - structures: list of dicts with 'formula', optionally 'crystal_structure',
          'lattice_constant', 'repeat', 'label' — for batch inference
        - input_file: container path to an extxyz file with one or more frames

    Model loading: tries torch.jit.load (TorchScript) first; falls back to
    torch.load(..., weights_only=False). KLAY MACE models are always saved via
    torch.save and will always use the fallback path. This is transparent.

    Args:
        model_path: Container path to the KIM model subdirectory (Name__MO_*_000/).
            Use the 'model_path_docker' key from train_mace result.
        formula: Chemical formula (e.g. "Si", "Fe", "NaCl", "Fe2O3").
            Only letters and digits — no spaces, brackets, or parentheses.
            Mutually exclusive with structures and input_file.
        crystal_structure: ASE bulk structure type ("diamond", "fcc", "bcc",
            "rocksalt", "wurtzite", "hcp", etc.) or "molecule" for gas-phase.
        lattice_constant: Optional lattice constant in Angstroms.
        repeat: Optional [nx, ny, nz] supercell repetitions applied to the ASE
            primitive cell. E.g. [2, 2, 2] gives 16 atoms for Si diamond (2-atom
            primitive cell). Only valid with formula.
        structures: List of structure dicts for batch inference. Each dict must
            have 'formula'; may also have 'crystal_structure', 'lattice_constant',
            'repeat', 'label'. Mutually exclusive with formula and input_file.
        input_file: Container path to an extxyz file (one or more frames).
            Mutually exclusive with formula and structures.
        calculations: Subset of ["energy", "forces", "stress", "relax"].
            Defaults to ["energy", "forces"]. "stress" returns a note, not an error.
        device: "cuda", "mps", or "cpu". Auto-detected if None.
        mode: "run" to execute and return results, "snippet" to return a
            copy-pasteable Python script. snippet mode only supports formula input.

    Returns:
        In run mode: dict with success, mode, frames (list of per-frame results),
            and output_file (extxyz path, multi-frame if batch).
        In snippet mode: dict with "snippet" key containing the Python script.
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        return {"success": False, "error": f"Model directory not found: {model_path}"}

    model_pt = model_dir / "model.pt"
    model_state_pt = model_dir / "model_state.pt"
    param_file = model_dir / "kliff_graph.param"
    if not model_pt.exists() and not model_state_pt.exists():
        return {"success": False, "error": f"Neither model.pt nor model_state.pt found in {model_path}"}
    if not param_file.exists():
        return {"success": False, "error": f"kliff_graph.param not found in {model_path}"}

    n_sources = sum([formula is not None, structures is not None, input_file is not None])
    if n_sources > 1:
        return {"success": False, "error": "'formula', 'structures', and 'input_file' are mutually exclusive."}
    if n_sources == 0:
        return {"success": False, "error": "One of 'formula', 'structures', or 'input_file' must be provided."}
    if input_file is not None and repeat is not None:
        return {"success": False, "error": "'repeat' is only valid with formula, not with input_file."}
    if mode == "snippet" and (input_file is not None or structures is not None):
        return {"success": False, "error": "snippet mode only supports formula/crystal_structure inputs."}

    if formula is not None:
        err = validate_structure_inputs(formula, crystal_structure)
        if err:
            return {"success": False, "error": err}

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
        snippet = _build_snippet(model_dir, formula, crystal_structure, lattice_constant, calcs, device)
        return {
            "success": True,
            "mode": "snippet",
            "model_path": str(model_dir),
            "snippet": snippet,
            "next_step": "Paste the snippet into a Python shell or Jupyter notebook.",
        }

    try:
        atoms_list, labels = _build_atom_frames(
            formula, crystal_structure, lattice_constant, repeat, structures, input_file
        )
    except Exception as e:
        return {"success": False, "error": str(e)}

    if formula is not None:
        output_tag = f"{formula}_{(crystal_structure or 'molecule').lower()}"
    elif input_file is not None:
        output_tag = Path(input_file).stem
    else:
        output_tag = "batch"

    return _run_calculations(model_dir, atoms_list, labels, calcs, device, output_tag)


def _parse_kliff_graph_param(param_file: Path) -> dict:
    """Parse kliff_graph.param to extract species, cutoff, n_layers."""
    lines = [l.strip() for l in param_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    result = {}
    idx = 0
    try:
        int(lines[idx]); idx += 1
        result["species"] = lines[idx].split(); idx += 1
        idx += 1
        result["cutoff"] = float(lines[idx]); idx += 1
        result["n_layers"] = int(lines[idx]); idx += 1
    except (IndexError, ValueError):
        pass
    return result


def _load_atoms_from_file(path: Path) -> tuple[list, list]:
    from ase.io import read
    frames = read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    labels = [a.info.get("config_type", f"frame_{i}") for i, a in enumerate(frames)]
    return frames, labels


def _build_atom_frames(formula, crystal_structure, lattice_constant, repeat, structures, input_file):
    if input_file is not None:
        return _load_atoms_from_file(Path(input_file))
    if structures is not None:
        frames, labels = [], []
        for i, spec in enumerate(structures):
            f = spec["formula"]
            cs = spec.get("crystal_structure")
            lc = spec.get("lattice_constant")
            r = spec.get("repeat")
            label = spec.get("label", f"frame_{i}")
            err = validate_structure_inputs(f, cs)
            if err:
                raise ValueError(err)
            frames.append(build_atoms(f, cs, lc, r))
            labels.append(label)
        return frames, labels
    atoms = build_atoms(formula, crystal_structure, lattice_constant, repeat)
    struct_tag = (crystal_structure or "molecule").lower()
    label = f"{formula}_{struct_tag}"
    if repeat:
        label += f"_{'x'.join(str(r) for r in repeat)}"
    return [atoms], [label]


def _write_extxyz(atoms_list: list, output_tag: str) -> Path:
    from ase.io import write
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = INFERENCE_DIR / f"{output_tag}_{timestamp}.extxyz"
    write(str(output_path), atoms_list, format="extxyz")
    return output_path


def _run_calculations(
    model_dir: Path,
    atoms_list: list,
    labels: list,
    calculations: list[str],
    device: str,
    output_tag: str,
) -> dict:
    try:
        import numpy as np
        import torch
        from torch_scatter import scatter_add
        from kliff.dataset import Configuration
        from kliff.transforms.configuration_transforms.graphs.generate_graph import RadialGraph
    except ImportError as e:
        return {"success": False, "error": f"Missing dependency: {e}. Install with pip install '.[full]'."}

    param_file = model_dir / "kliff_graph.param"
    params = _parse_kliff_graph_param(param_file)
    if not params.get("species") or not params.get("cutoff"):
        return {"success": False, "error": f"Could not parse {param_file}"}

    try:
        model_state_pt = model_dir / "model_state.pt"
        yaml_path = model_dir / "mace_model.yaml"
        if model_state_pt.exists() and yaml_path.exists():
            import yaml as _yaml
            from omegaconf import OmegaConf
            from klay.builder import build_model
            with open(yaml_path) as f:
                cfg_dict = _yaml.safe_load(f)
            model = build_model(OmegaConf.create(cfg_dict))
            state_dict = torch.load(str(model_state_pt), map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            model = model.to(device)
        else:
            try:
                model = torch.jit.load(str(model_dir / "model.pt"), map_location=device)
            except Exception:
                model = torch.load(str(model_dir / "model.pt"), map_location=device, weights_only=False)
        model.eval()
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32
    except Exception as e:
        return {"success": False, "error": f"Failed to load model: {e}"}

    transform = RadialGraph(
        species=params["species"],
        cutoff=params["cutoff"],
        n_layers=params["n_layers"],
    )

    frames_results = []
    result_atoms_list = []
    for atoms, label in zip(atoms_list, labels):
        frame_result = {"label": label}
        try:
            n_orig = len(atoms)
            if "relax" in calculations:
                from ase.optimize import BFGS
                calc = _KliffInlineCalculator(model, transform, params, device, n_orig)
                atoms_work = atoms.copy()
                atoms_work.calc = calc
                opt = BFGS(atoms_work, logfile=None)
                converged = opt.run(fmax=0.01, steps=500)
                frame_result["relaxation_converged"] = converged
                atoms = atoms_work

            cell = atoms.cell.array
            species_list = list(atoms.get_chemical_symbols())
            coords_np = atoms.get_positions()
            pbc = list(atoms.get_pbc())
            config = Configuration(
                cell=cell, species=species_list, coords=coords_np, PBC=pbc,
                energy=0.0, forces=np.zeros((len(species_list), 3)),
            )
            graph = transform(config)
            dev = torch.device(device)
            coords_t = graph.coords.clone().detach().to(model_dtype).to(dev).requires_grad_(True)
            species_t = graph.species.to(dev)
            edge_index_t = graph.edge_index0.to(dev)
            contributions_t = graph.contributions.to(dev)
            images_t = graph.images.to(dev)

            energy_t = model(
                species=species_t, coords=coords_t,
                edge_index0=edge_index_t, contributions=contributions_t,
            )
            if "energy" in calculations or "relax" in calculations:
                frame_result["energy_eV"] = float(energy_t.sum().item())
            if "forces" in calculations or "relax" in calculations:
                (grad,) = torch.autograd.grad(energy_t.sum(), coords_t, create_graph=False)
                forces_t = -scatter_add(grad, images_t, dim=0)[:n_orig]
                frame_result["forces_eV_per_Ang"] = forces_t.detach().cpu().tolist()
            if "stress" in calculations:
                frame_result["stress_note"] = "Stress calculation requires PBC and is not yet implemented."
            if "relax" in calculations:
                frame_result["relaxed_positions"] = atoms.get_positions().tolist()
                frame_result["relaxed_cell"] = atoms.get_cell().tolist()
        except Exception as e:
            frame_result["error"] = str(e)
        frames_results.append(frame_result)
        result_atoms_list.append(atoms)

    result = {"success": True, "mode": "run", "frames": frames_results}
    try:
        output_path = _write_extxyz(result_atoms_list, output_tag)
        result["output_file"] = str(output_path)
    except Exception as e:
        result["output_file_warning"] = f"Results computed but extxyz write failed: {e}"
    return result


def _expand_coords_for_images(coords: "torch.Tensor", images: "torch.Tensor") -> "torch.Tensor":
    """Re-index coords by images to get expanded (ghost-atom) coordinate tensor."""
    return coords[images]


class _KliffInlineCalculator:
    """Minimal ASE-compatible calculator wrapping a KLAY TorchScript model."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, model, transform, params, device, n_orig):
        import torch
        self.model = model
        self.transform = transform
        self.params = params
        self.device = device
        self.n_orig = n_orig
        self.results = {}
        try:
            self.model_dtype = next(model.parameters()).dtype
        except StopIteration:
            self.model_dtype = torch.float32

    def calculate(self, atoms, properties=None, system_changes=None):
        import numpy as np
        import torch
        from torch_scatter import scatter_add
        from kliff.dataset import Configuration

        dev = torch.device(self.device)
        cell = atoms.cell.array
        species_list = list(atoms.get_chemical_symbols())
        coords_np = atoms.get_positions()
        pbc = list(atoms.get_pbc())
        config = Configuration(
            cell=cell, species=species_list, coords=coords_np, PBC=pbc,
            energy=0.0, forces=np.zeros((len(species_list), 3)),
        )
        graph = self.transform(config)
        coords_t = graph.coords.clone().detach().to(self.model_dtype).to(dev).requires_grad_(True)
        species_t = graph.species.to(dev)
        edge_index_t = graph.edge_index0.to(dev)
        contributions_t = graph.contributions.to(dev)
        images_t = graph.images.to(dev)

        energy_t = self.model(
            species=species_t, coords=coords_t,
            edge_index0=edge_index_t, contributions=contributions_t,
        )
        (grad,) = torch.autograd.grad(energy_t.sum(), coords_t, create_graph=False)
        forces_t = -scatter_add(grad, images_t, dim=0)[:self.n_orig]
        self.results = {
            "energy": float(energy_t.sum().item()),
            "forces": forces_t.detach().cpu().numpy(),
        }

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.results.get("energy", 0.0)

    def get_forces(self, atoms=None):
        import numpy as np
        return self.results.get("forces", np.zeros((self.n_orig, 3)))


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


def _build_snippet(
    model_dir: Path,
    formula: str,
    crystal_structure: str | None,
    lattice_constant: float | None,
    calculations: list[str],
    device: str,
) -> str:
    param_file = model_dir / "kliff_graph.param"
    species_val = "..."
    cutoff_val = "..."
    n_layers_val = 1
    if param_file.exists():
        parsed = _parse_kliff_graph_param(param_file)
        if parsed.get("species"):
            species_val = repr(parsed["species"])
        if parsed.get("cutoff"):
            cutoff_val = repr(parsed["cutoff"])
        if parsed.get("n_layers"):
            n_layers_val = parsed["n_layers"]
    lines = [
        "import numpy as np",
        "import torch",
        "from torch_scatter import scatter_add",
        "from kliff.dataset import Configuration",
        "from kliff.transforms.configuration_transforms.graphs.generate_graph import RadialGraph",
        _structure_import(crystal_structure),
        "",
        _structure_creation(formula, crystal_structure, lattice_constant),
        "",
        f"model_dir = {str(model_dir)!r}",
        "try:",
        "    model = torch.jit.load(f\"{model_dir}/model.pt\")",
        "except Exception:",
        "    model = torch.load(f\"{model_dir}/model.pt\", weights_only=False)",
        "model.eval()",
        "",
        f"species = {species_val}",
        f"cutoff = {cutoff_val}",
        f"transform = RadialGraph(species=species, cutoff=cutoff, n_layers={n_layers_val})",
        "",
        "config = Configuration(",
        "    cell=atoms.cell.array, species=list(atoms.get_chemical_symbols()),",
        "    coords=atoms.get_positions(), PBC=list(atoms.get_pbc()),",
        "    energy=0.0, forces=np.zeros((len(atoms), 3)),",
        ")",
        "graph = transform(config)",
        f"dev = torch.device({device!r})",
        "model_dtype = next(model.parameters()).dtype",
        "coords = graph.coords.clone().detach().to(model_dtype).to(dev).requires_grad_(True)",
        "energy = model(",
        "    species=graph.species.to(dev),",
        "    coords=coords,",
        "    edge_index0=graph.edge_index0.to(dev),",
        "    contributions=graph.contributions.to(dev),",
        ")",
        "print(f\"Energy: {energy.sum().item():.4f} eV\")",
    ]
    if "forces" in calculations:
        lines += [
            "(grad,) = torch.autograd.grad(energy.sum(), coords)",
            "forces = -scatter_add(grad, graph.images.to(dev), dim=0)[:len(atoms)]",
            "print(f\"Forces (eV/Å):\\n{forces.detach().cpu().numpy()}\")",
        ]
    return "\n".join(lines)
