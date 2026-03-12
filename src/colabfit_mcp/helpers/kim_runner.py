from pathlib import Path

from ase.calculators.calculator import Calculator, all_changes

_KIMVV_TEST_DRIVERS = {
    "EquilibriumCrystalStructure": {
        "description": "Equilibrium lattice parameters and cohesive energy",
        "properties": ["lattice-constant", "cohesive-energy"],
        "requires_crystal": True,
    },
    "ElasticConstantsCrystal": {
        "description": "Full elastic constants tensor at zero temperature",
        "properties": ["elastic-constants"],
        "requires_crystal": True,
    },
    "CrystalStructureAndEnergyVsPressure": {
        "description": "Crystal structure and energy as a function of pressure",
        "properties": ["energy-vs-pressure"],
        "requires_crystal": True,
    },
    "GroundStateCrystalStructure": {
        "description": "Lowest energy crystal structure among candidates",
        "properties": ["ground-state-structure"],
        "requires_crystal": True,
    },
    "VacancyFormationEnergyRelaxationVolumeCrystal": {
        "description": "Vacancy formation energy and relaxation volume",
        "properties": ["vacancy-formation-energy", "relaxation-volume"],
        "requires_crystal": True,
    },
    "ClusterEnergyAndForces": {
        "description": (
            "BFGS relaxation of an atomic cluster in a non-periodic box. "
            "Returns relaxed geometry, energy, and forces. "
            "Use for models trained on molecular datasets."
        ),
        "properties": ["energy", "atomic-forces", "relaxed-positions"],
        "requires_crystal": False,
    },
}


def parse_model_params(model_dir: Path) -> dict:
    """Parse kliff_graph.param → {"species": [...], "cutoff": float, "n_layers": int}."""
    param_file = model_dir / "kliff_graph.param"
    if not param_file.exists():
        return {}
    lines = [
        l.strip()
        for l in param_file.read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    result = {}
    idx = 0
    try:
        int(lines[idx])
        idx += 1
        result["species"] = lines[idx].split()
        idx += 1
        idx += 1
        result["cutoff"] = float(lines[idx])
        idx += 1
        result["n_layers"] = int(lines[idx])
    except (IndexError, ValueError):
        pass
    return result


class KlayASECalculator(Calculator):
    """ASE Calculator subclass wrapping a KLAY model for use with kimvv test drivers."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model, transform, params: dict, device: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.transform = transform
        self.params = params
        self.device = device
        import torch
        try:
            self.model_dtype = next(model.parameters()).dtype
        except StopIteration:
            self.model_dtype = torch.float32

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        import numpy as np
        import torch
        from torch_scatter import scatter_add
        from kliff.dataset import Configuration

        super().calculate(atoms, properties, system_changes)
        if properties is None:
            properties = self.implemented_properties

        n_orig = len(atoms)
        dev = torch.device(self.device)
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
            forces=np.zeros((n_orig, 3)),
        )
        graph = self.transform(config)
        coords_t = (
            graph.coords.clone().detach().to(self.model_dtype).to(dev).requires_grad_(True)
        )
        species_t = graph.species.to(dev)
        edge_index_t = graph.edge_index0.to(dev)
        contributions_t = graph.contributions.to(dev)
        images_t = graph.images.to(dev)

        energy_t = self.model(
            species=species_t,
            coords=coords_t,
            edge_index0=edge_index_t,
            contributions=contributions_t,
        )
        (grad,) = torch.autograd.grad(energy_t.sum(), coords_t, create_graph=False)
        forces_t = -scatter_add(grad, images_t, dim=0)[:n_orig]
        energy = float(energy_t.sum().item())
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces_t.detach().cpu().numpy(),
        }
        if "stress" in properties:
            saved = dict(self.results)
            self.results["stress"] = self.calculate_numerical_stress(atoms)
            self.results.update(saved)


def run_cluster_energy_and_forces(calc: KlayASECalculator, atoms) -> dict:
    """BFGS-relax atoms in a non-periodic box; return energy, forces, relaxed positions."""
    from ase.optimize import BFGS

    atoms = atoms.copy()
    atoms.pbc = False
    atoms.center(vacuum=10.0)
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.05)
    return {
        "energy_eV": float(atoms.get_potential_energy()),
        "forces_eV_per_Ang": atoms.get_forces().tolist(),
        "relaxed_positions": atoms.get_positions().tolist(),
        "n_atoms": len(atoms),
    }


def load_klay_calculator(model_dir: Path, device: str) -> KlayASECalculator:
    """Load KLAY model from model_dir and wrap in KlayASECalculator."""
    import torch
    from kliff.transforms.configuration_transforms.graphs.generate_graph import RadialGraph

    params = parse_model_params(model_dir)
    if not params.get("species") or not params.get("cutoff"):
        raise ValueError(f"Could not parse kliff_graph.param in {model_dir}")

    model_state_pt = model_dir / "model_state.pt"
    yaml_path = model_dir / "mace_model.yaml"
    try:
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
                model = torch.load(
                    str(model_dir / "model.pt"), map_location=device, weights_only=False
                )
    except RuntimeError as e:
        if "device-side assert" in str(e) or "CUDA error" in str(e):
            raise RuntimeError(
                "CUDA device-side assert triggered — the CUDA context is now corrupted. "
                "All further GPU operations in this session will fail. "
                "Restart the MCP server (restart Claude Code) to recover."
            ) from e
        raise
    model.eval()

    transform = RadialGraph(
        species=params["species"],
        cutoff=params["cutoff"],
        n_layers=params.get("n_layers", 1),
    )
    return KlayASECalculator(model=model, transform=transform, params=params, device=device)


def check_element_compatibility(model_elements: list[str], formula: str) -> str | None:
    """Return error string if formula contains elements not in model_elements, else None."""
    from ase.formula import Formula

    try:
        formula_elements = set(Formula(formula).count().keys())
    except Exception:
        return f"Could not parse formula {formula!r}."
    unknown = formula_elements - set(model_elements)
    if unknown:
        return (
            f"Formula {formula!r} contains elements {sorted(unknown)} "
            f"not in model elements {model_elements}."
        )
    return None
