COORDS_KEY = "positions"
SPECIES_KEY = "atomic_numbers"
PBC_KEY = "pbc"
CELL_KEY = "cell"
ENERGY_KEY = "energy"
FORCES_KEY = "atomic_forces"


def download_from_huggingface(
    hf_id: str,
    split: str,
    n_configs: int | None = None,
    cache_dir: str | None = None,
):
    """Load a HuggingFace dataset split and return only the training-relevant columns.

    The full split is cached locally by the datasets library. If n_configs is given,
    the returned dataset is truncated to that many rows; the on-disk cache is not
    affected.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_id, split=split, cache_dir=cache_dir)
    if n_configs is not None:
        ds = ds.select(range(min(n_configs, len(ds))))

    present = set(ds.column_names)
    cols = [COORDS_KEY, SPECIES_KEY, PBC_KEY, CELL_KEY]
    for optional in [ENERGY_KEY, FORCES_KEY]:
        if optional in present:
            cols.append(optional)

    return ds.select_columns(cols)


def analyze_hf_dataset(ds) -> dict:
    """Return a training-suitability summary for a HuggingFace dataset.

    The returned dict has the same keys as kliff_utils.analyze_configs so all
    downstream callers (check_local_datasets, resolve_dataset, _suggest_next_step)
    work without modification.
    """
    n = len(ds)
    if n == 0:
        return {
            "elements": [],
            "n_configs": 0,
            "n_atoms_total": 0,
            "has_energy": False,
            "has_forces": False,
            "has_stress": False,
            "suitable_for_training": False,
        }

    col_names = set(ds.column_names)
    first = ds[0]
    has_energy = ENERGY_KEY in col_names and first.get(ENERGY_KEY) is not None
    has_forces = FORCES_KEY in col_names and first.get(FORCES_KEY) is not None

    from ase.data import chemical_symbols

    sample_size = min(100, n)
    elements: set[str] = set()
    n_atoms_sample = 0
    for atomic_nums in ds.select(range(sample_size))[SPECIES_KEY]:
        for z in atomic_nums:
            elements.add(chemical_symbols[int(z)])
        n_atoms_sample += len(atomic_nums)

    avg_atoms = n_atoms_sample / sample_size if sample_size > 0 else 0

    return {
        "elements": sorted(elements),
        "n_configs": n,
        "n_atoms_total": int(avg_atoms * n),
        "has_energy": has_energy,
        "has_forces": has_forces,
        "has_stress": False,
        "suitable_for_training": has_energy and has_forces and n > 0,
    }
