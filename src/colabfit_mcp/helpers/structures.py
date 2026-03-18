import re

VALID_CRYSTAL_STRUCTURES = {
    "sc", "fcc", "bcc", "tetragonal", "bct", "hcp", "rhombohedral",
    "orthorhombic", "mcl", "diamond", "zincblende", "rocksalt",
    "cesiumchloride", "fluorite", "wurtzite", "molecule",
}

_FORMULA_RE = re.compile(r"^[A-Za-z0-9]+$")


def validate_structure_inputs(formula: str | None, crystal_structure: str | None) -> str | None:
    if formula is None:
        return "formula is required."
    if not _FORMULA_RE.fullmatch(formula):
        return f"Invalid formula {formula!r}: only letters and digits are allowed."
    if crystal_structure is not None and crystal_structure.lower() not in VALID_CRYSTAL_STRUCTURES:
        return (
            f"Invalid crystal_structure {crystal_structure!r}. "
            f"Valid options: {sorted(VALID_CRYSTAL_STRUCTURES)}"
        )
    return None


def build_atoms(
    formula: str,
    crystal_structure: str | None,
    lattice_constant: float | None,
    repeat: list[int] | None = None,
):
    if crystal_structure and crystal_structure.lower() != "molecule":
        from ase.build import bulk
        kwargs = {}
        if lattice_constant is not None:
            kwargs["a"] = lattice_constant
        atoms = bulk(formula, crystal_structure, **kwargs)
    else:
        from ase.build import molecule
        atoms = molecule(formula)
    if repeat is not None:
        atoms = atoms.repeat(repeat)
    return atoms
