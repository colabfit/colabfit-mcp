_CRYSTAL_STRUCTURE_INFO = {
    "sc": {
        "n_formula_atoms": 1,
        "formula_pattern": "single element only (e.g. 'Po')",
        "lattice_constant_required": False,
        "note": "auto-resolved for most elements; required if element has no sc reference",
        "example_formula": "Po",
        "example_lattice_constant": None,
    },
    "fcc": {
        "n_formula_atoms": 1,
        "formula_pattern": "single element only (e.g. 'Al', 'Cu', 'Ni', 'Au', 'Pt')",
        "lattice_constant_required": False,
        "note": "auto-resolved for most FCC metals",
        "example_formula": "Al",
        "example_lattice_constant": None,
    },
    "bcc": {
        "n_formula_atoms": 1,
        "formula_pattern": "single element only (e.g. 'Fe', 'W', 'Mo', 'Cr', 'V')",
        "lattice_constant_required": False,
        "note": "auto-resolved for most BCC metals",
        "example_formula": "Fe",
        "example_lattice_constant": None,
    },
    "hcp": {
        "n_formula_atoms": 1,
        "formula_pattern": "single element only (e.g. 'Ti', 'Mg', 'Zn', 'Co', 'Zr')",
        "lattice_constant_required": False,
        "note": "auto-resolved for most HCP metals; c/a defaults to ideal sqrt(8/3)",
        "example_formula": "Ti",
        "example_lattice_constant": None,
    },
    "diamond": {
        "n_formula_atoms": 1,
        "formula_pattern": "single element only (e.g. 'Si', 'Ge', 'C')",
        "lattice_constant_required": False,
        "note": "auto-resolved for Si, Ge, C",
        "example_formula": "Si",
        "example_lattice_constant": None,
    },
    "tetragonal": {
        "n_formula_atoms": 1,
        "formula_pattern": (
            "single element ONLY — NEVER use for multi-element formulas like TiO2. "
            "TiO2/rutile is NOT tetragonal in ASE — use fluorite instead."
        ),
        "lattice_constant_required": True,
        "note": "requires a; most elements also need c (pass as second positional after a). "
                "Very few elements have reference data for this structure.",
        "example_formula": "In",
        "example_lattice_constant": 3.25,
    },
    "bct": {
        "n_formula_atoms": 1,
        "formula_pattern": "single element ONLY (body-centered tetragonal, e.g. 'In', 'Sn')",
        "lattice_constant_required": True,
        "note": "requires a; most elements also need c",
        "example_formula": "In",
        "example_lattice_constant": 3.25,
    },
    "rhombohedral": {
        "n_formula_atoms": 1,
        "formula_pattern": "single element ONLY (e.g. 'Bi', 'As', 'Sb', 'Hg')",
        "lattice_constant_required": False,
        "note": "auto-resolved for elements with rhombohedral reference state",
        "example_formula": "Bi",
        "example_lattice_constant": None,
    },
    "orthorhombic": {
        "n_formula_atoms": 1,
        "formula_pattern": "single element ONLY (e.g. 'S', 'Ga', 'P')",
        "lattice_constant_required": False,
        "note": "auto-resolved for elements with orthorhombic reference state",
        "example_formula": "S",
        "example_lattice_constant": None,
    },
    "mcl": {
        "n_formula_atoms": 1,
        "formula_pattern": "single element ONLY (monoclinic, e.g. 'Se', 'P')",
        "lattice_constant_required": False,
        "note": "auto-resolved for elements with mcl reference state",
        "example_formula": "Se",
        "example_lattice_constant": None,
    },
    "rocksalt": {
        "n_formula_atoms": 2,
        "formula_pattern": (
            "exactly two elements in 1:1 stoichiometric ratio "
            "(e.g. 'NaCl', 'MgO', 'LiF', 'FeO', 'CoO'). "
            "Formula must parse to exactly 2 atomic symbols."
        ),
        "lattice_constant_required": True,
        "note": "ALWAYS requires lattice_constant — no built-in reference data for compounds",
        "example_formula": "NaCl",
        "example_lattice_constant": 5.64,
    },
    "cesiumchloride": {
        "n_formula_atoms": 2,
        "formula_pattern": (
            "exactly two elements in 1:1 ratio (e.g. 'CsCl', 'TlBr', 'BeCu'). "
            "Formula must parse to exactly 2 atomic symbols."
        ),
        "lattice_constant_required": True,
        "note": "ALWAYS requires lattice_constant",
        "example_formula": "CsCl",
        "example_lattice_constant": 4.12,
    },
    "zincblende": {
        "n_formula_atoms": 2,
        "formula_pattern": (
            "exactly two elements in 1:1 ratio "
            "(e.g. 'GaAs', 'ZnS', 'SiC', 'InP', 'AlAs'). "
            "Formula must parse to exactly 2 atomic symbols."
        ),
        "lattice_constant_required": True,
        "note": "ALWAYS requires lattice_constant",
        "example_formula": "GaAs",
        "example_lattice_constant": 5.65,
    },
    "wurtzite": {
        "n_formula_atoms": 2,
        "formula_pattern": (
            "exactly two elements in 1:1 ratio "
            "(e.g. 'ZnO', 'GaN', 'CdS', 'AlN', 'BN'). "
            "Formula must parse to exactly 2 atomic symbols."
        ),
        "lattice_constant_required": True,
        "note": "ALWAYS requires lattice_constant",
        "example_formula": "ZnO",
        "example_lattice_constant": 3.25,
    },
    "fluorite": {
        "n_formula_atoms": 3,
        "formula_pattern": (
            "exactly two elements in 1:2 ratio, formula AB2 "
            "(e.g. 'TiO2', 'CaF2', 'SnO2', 'UO2', 'CeO2'). "
            "Formula must parse to exactly 3 atomic symbols (1 A + 2 B)."
        ),
        "lattice_constant_required": True,
        "note": (
            "ALWAYS requires lattice_constant. "
            "THIS is the correct structure for TiO2/rutile, SnO2/cassiterite — "
            "NOT 'tetragonal'. Tetragonal in ASE is single-element only."
        ),
        "example_formula": "TiO2",
        "example_lattice_constant": 4.59,
    },
}

_CRYSTAL_STRUCTURE_EXAMPLES = {
    "Si":   {"crystal_structure": "diamond",       "lattice_constant": None},
    "Ge":   {"crystal_structure": "diamond",       "lattice_constant": None},
    "C":    {"crystal_structure": "diamond",       "lattice_constant": None},
    "Fe":   {"crystal_structure": "bcc",           "lattice_constant": None},
    "W":    {"crystal_structure": "bcc",           "lattice_constant": None},
    "Mo":   {"crystal_structure": "bcc",           "lattice_constant": None},
    "Al":   {"crystal_structure": "fcc",           "lattice_constant": None},
    "Cu":   {"crystal_structure": "fcc",           "lattice_constant": None},
    "Ni":   {"crystal_structure": "fcc",           "lattice_constant": None},
    "Au":   {"crystal_structure": "fcc",           "lattice_constant": None},
    "Ti":   {"crystal_structure": "hcp",           "lattice_constant": None},
    "Mg":   {"crystal_structure": "hcp",           "lattice_constant": None},
    "Zn":   {"crystal_structure": "hcp",           "lattice_constant": None},
    "NaCl": {"crystal_structure": "rocksalt",      "lattice_constant": 5.64},
    "MgO":  {"crystal_structure": "rocksalt",      "lattice_constant": 4.21},
    "LiF":  {"crystal_structure": "rocksalt",      "lattice_constant": 4.02},
    "CsCl": {"crystal_structure": "cesiumchloride","lattice_constant": 4.12},
    "GaAs": {"crystal_structure": "zincblende",    "lattice_constant": 5.65},
    "SiC":  {"crystal_structure": "zincblende",    "lattice_constant": 4.36},
    "ZnO":  {"crystal_structure": "wurtzite",      "lattice_constant": 3.25},
    "GaN":  {"crystal_structure": "wurtzite",      "lattice_constant": 3.19},
    "TiO2": {"crystal_structure": "fluorite",      "lattice_constant": 4.59},
    "CaF2": {"crystal_structure": "fluorite",      "lattice_constant": 5.46},
    "SnO2": {"crystal_structure": "fluorite",      "lattice_constant": 4.74},
}

_ASE_MOLECULE_NAMES = [
    "H2O", "CH4", "CO2", "NH3", "H2", "O2", "N2", "CO", "NO",
    "C2H2", "C2H4", "C2H6", "CH3OH", "C6H6", "HF", "HCl",
    "NaCl", "SO2", "H2S", "PH3",
]
