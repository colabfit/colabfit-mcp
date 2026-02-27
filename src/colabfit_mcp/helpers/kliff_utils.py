def fix_species_types(configs) -> None:
    """Convert atomic-number species to element symbols in-place.

    Dataset.from_huggingface stores species as ints (e.g. 42 for Mo);
    RadialGraph's C extension get_complete_graph expects symbol strings.
    """
    from ase.data import chemical_symbols
    for config in configs:
        if config._species and isinstance(config._species[0], (int, float)):
            config._species = [chemical_symbols[int(s)] for s in config._species]


def analyze_configs(configs) -> dict:
    """Analyze KLIFF Configuration objects for training suitability.

    config.species may be element symbols (str) or atomic numbers (int) depending
    on how the Dataset was loaded; both are handled via ase.data.chemical_symbols.
    config._energy and config._forces are used directly (the .energy/.forces
    properties raise ConfigurationError when None).
    """
    from ase.data import chemical_symbols

    elements = set()
    n_atoms_total = 0
    has_energy = False
    has_forces = False
    for config in configs:
        for s in config.species:
            elements.add(chemical_symbols[int(s)] if isinstance(s, (int, float)) else str(s))
        n_atoms_total += len(config.species)
        if config._energy is not None:
            has_energy = True
        if config._forces is not None:
            has_forces = True
    n = len(configs)
    return {
        "elements": sorted(elements),
        "n_configs": n,
        "n_atoms_total": n_atoms_total,
        "has_energy": has_energy,
        "has_forces": has_forces,
        "suitable_for_training": has_energy and has_forces and n > 0,
    }
