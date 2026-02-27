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
    """Run ASE calculations with a trained KLAY/KLIFF KIM model, or generate a code snippet.

    Builds an ASE Atoms object from the specified structure, loads the KLAY model
    from the KIM model directory (produced by train_mace), and either executes the
    requested calculations immediately (mode='run') or returns a copy-pasteable Python
    script (mode='snippet').

    IMPORTANT: model_path must be the KIM model *subdirectory* returned as
    'model_path' by train_mace (format: Name__MO_000000000000_000/), not the
    parent model_dir. The subdirectory must contain model.pt and kliff_graph.param.

    IMPORTANT: All paths are inside the Docker container filesystem.

    Model loading: tries torch.jit.load (TorchScript) first; falls back to
    torch.load(..., weights_only=False). Due to a TorchScript incompatibility in
    OneHotAtomEncoding, KLAY MACE models are always saved via torch.save and will
    always use the fallback path. This is transparent — no action required.

    Args:
        model_path: Container path to the KIM model subdirectory (Name__MO_*_000/).
            Use the 'model_path' key from train_mace result, not 'model_dir'.
        formula: Chemical formula (e.g. "Si", "Fe", "NaCl", "Fe2O3").
            Only letters and digits are valid — no spaces, brackets, or parentheses.
        crystal_structure: ASE bulk structure type ("diamond", "fcc", "bcc",
            "rocksalt", "wurtzite", "hcp", etc.) or "molecule" for gas-phase.
            Must be one of the ASE-recognized names; see valid list in error messages.
        lattice_constant: Optional lattice constant in Angstroms passed to bulk().
            If omitted, ASE uses its built-in default for the element.
        calculations: Subset of ["energy", "forces", "stress", "relax"].
            Defaults to ["energy", "forces"]. Including "relax" automatically
            computes energy and forces too — no need to list them separately.
            "stress" is accepted but not yet implemented (returns a note, not an error).
        device: "cuda", "mps", or "cpu". Auto-detected if None.
        mode: "run" to execute and return results, "snippet" to return a
            copy-pasteable Python script using torch + kliff directly.

    Returns:
        In run mode: dict with energy_eV, forces_eV_per_Ang, relaxed_positions,
            relaxed_cell (if relax requested), and output_file (extxyz path).
        In snippet mode: dict with "snippet" key containing the Python script.
    """
    model_dir = Path(model_path)
    if not model_dir.exists():
        return {"success": False, "error": f"Model directory not found: {model_path}"}

    model_pt = model_dir / "model.pt"
    param_file = model_dir / "kliff_graph.param"
    if not model_pt.exists():
        return {"success": False, "error": f"model.pt not found in {model_path}"}
    if not param_file.exists():
        return {"success": False, "error": f"kliff_graph.param not found in {model_path}"}

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
        snippet = _build_snippet(model_dir, formula, crystal_structure, lattice_constant, calcs, device)
        return {
            "success": True,
            "mode": "snippet",
            "model_path": str(model_dir),
            "snippet": snippet,
            "next_step": "Paste the snippet into a Python shell or Jupyter notebook.",
        }

    return _run_calculations(model_dir, formula, crystal_structure, lattice_constant, calcs, device)


def _parse_kliff_graph_param(param_file: Path) -> dict:
    """Parse kliff_graph.param to extract species, cutoff, n_layers."""
    lines = [l.strip() for l in param_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    result = {}
    idx = 0
    try:
        n_species = int(lines[idx]); idx += 1
        result["species"] = lines[idx].split(); idx += 1
        # "Graph"
        idx += 1
        result["cutoff"] = float(lines[idx]); idx += 1
        result["n_layers"] = int(lines[idx]); idx += 1
    except (IndexError, ValueError):
        pass
    return result


def _build_atoms(formula: str, crystal_structure: str | None, lattice_constant: float | None):
    if crystal_structure and crystal_structure.lower() != "molecule":
        from ase.build import bulk
        kwargs = {}
        if lattice_constant is not None:
            kwargs["a"] = lattice_constant
        return bulk(formula, crystal_structure, **kwargs)
    from ase.build import molecule
    return molecule(formula)


def _write_extxyz(atoms, formula: str, crystal_structure: str | None) -> Path:
    from ase.io import write

    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    struct_tag = (crystal_structure or "molecule").lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = INFERENCE_DIR / f"{formula}_{struct_tag}_{timestamp}.extxyz"
    write(str(output_path), atoms, format="extxyz")
    return output_path


def _run_calculations(
    model_dir: Path,
    formula: str,
    crystal_structure: str | None,
    lattice_constant: float | None,
    calculations: list[str],
    device: str,
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
        atoms = _build_atoms(formula, crystal_structure, lattice_constant)
    except Exception as e:
        return {"success": False, "error": f"Failed to build structure: {e}"}

    try:
        try:
            model = torch.jit.load(str(model_dir / "model.pt"), map_location=device)
        except Exception:
            model = torch.load(str(model_dir / "model.pt"), map_location=device, weights_only=False)
        model.eval()
    except Exception as e:
        return {"success": False, "error": f"Failed to load model: {e}"}

    try:
        transform = RadialGraph(
            species=params["species"],
            cutoff=params["cutoff"],
            n_layers=params["n_layers"],
        )
        cell = atoms.cell.array
        species_list = list(atoms.get_chemical_symbols())
        coords_np = atoms.get_positions()
        pbc = list(atoms.get_pbc())
        config = Configuration(
            cell=cell,
            species=species_list,
            coords=coords_np,
            PBC=pbc,
            energy=0.0,
            forces=np.zeros((len(species_list), 3)),
        )
        graph = transform(config)
    except Exception as e:
        return {"success": False, "error": f"Failed to build graph: {e}"}

    results = {}
    try:
        dev = torch.device(device)
        coords_t = graph.coords.clone().detach().to(torch.float64).to(dev).requires_grad_(True)
        species_t = graph.species.to(dev)
        edge_index_t = graph.edge_index0.to(dev)
        contributions_t = graph.contributions.to(dev)
        images_t = graph.images.to(dev)

        n_orig = len(atoms)

        if "relax" in calculations:
            from ase.optimize import BFGS
            calc = _KliffInlineCalculator(
                model, transform, params, device, n_orig
            )
            atoms_relax = atoms.copy()
            atoms_relax.calc = calc
            opt = BFGS(atoms_relax, logfile=None)
            converged = opt.run(fmax=0.01, steps=500)
            results["relaxation_converged"] = converged
            atoms = atoms_relax
            coords_t = torch.tensor(
                atoms.get_positions(), dtype=torch.float64, device=dev, requires_grad=True
            )
            coords_t_expanded = _expand_coords_for_images(coords_t, images_t)
            energy_t = model(
                species=species_t,
                coords=coords_t_expanded,
                edge_index0=edge_index_t,
                contributions=contributions_t,
            )
        else:
            energy_t = model(
                species=species_t,
                coords=coords_t,
                edge_index0=edge_index_t,
                contributions=contributions_t,
            )

        if "energy" in calculations or "relax" in calculations:
            results["energy_eV"] = float(energy_t.sum().item())

        if "forces" in calculations or "relax" in calculations:
            (grad,) = torch.autograd.grad(energy_t.sum(), coords_t, create_graph=False)
            forces_t = -scatter_add(grad, images_t, dim=0)[:n_orig]
            results["forces_eV_per_Ang"] = forces_t.detach().cpu().tolist()

        if "stress" in calculations:
            results["stress_note"] = "Stress calculation requires PBC and is not yet implemented."

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


def _expand_coords_for_images(coords: "torch.Tensor", images: "torch.Tensor") -> "torch.Tensor":
    """Re-index coords by images to get expanded (ghost-atom) coordinate tensor."""
    return coords[images]


class _KliffInlineCalculator:
    """Minimal ASE-compatible calculator wrapping a KLAY TorchScript model."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, model, transform, params, device, n_orig):
        self.model = model
        self.transform = transform
        self.params = params
        self.device = device
        self.n_orig = n_orig
        self.results = {}

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
        coords_t = graph.coords.clone().detach().to(torch.float64).to(dev).requires_grad_(True)
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
        "# parse kliff_graph.param for species/cutoff",
        "# (see colabfit_mcp.tools.use_model._parse_kliff_graph_param)",
        "transform = RadialGraph(species=..., cutoff=..., n_layers=1)  # fill from kliff_graph.param",
        "",
        "config = Configuration(",
        "    cell=atoms.cell.array, species=list(atoms.get_chemical_symbols()),",
        "    coords=atoms.get_positions(), PBC=list(atoms.get_pbc()),",
        "    energy=0.0, forces=np.zeros((len(atoms), 3)),",
        ")",
        "graph = transform(config)",
        f"dev = torch.device({device!r})",
        "coords = graph.coords.clone().detach().to(torch.float64).to(dev).requires_grad_(True)",
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
