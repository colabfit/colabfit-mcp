from pathlib import Path


def extract_elements(xyz_path: Path) -> list[str]:
    elements = set()
    with open(xyz_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                elem = parts[0]
                if elem.isalpha() and len(elem) <= 2:
                    elements.add(elem)
    return sorted(elements)


def analyze_xyz(xyz_path: Path) -> dict:
    """Analyze a single XYZ/extxyz file for training suitability."""
    elements = set()
    config_count = 0
    has_energy = False
    has_forces = False
    has_stress = False
    atom_count = 0

    with open(xyz_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue

        config_count += 1
        atom_count += n_atoms

        if i + 1 < len(lines):
            comment = lines[i + 1]
            if "energy" in comment.lower():
                has_energy = True
            if "stress" in comment.lower():
                has_stress = True
            if "forces" in comment.lower():
                has_forces = True

        for j in range(i + 2, min(i + 2 + n_atoms, len(lines))):
            parts = lines[j].strip().split()
            if parts and parts[0].isalpha() and len(parts[0]) <= 2:
                elements.add(parts[0])
            if len(parts) > 4:
                has_forces = True

        i += 2 + n_atoms

    return {
        "elements": sorted(elements),
        "n_configs": config_count,
        "n_atoms_total": atom_count,
        "has_energy": has_energy,
        "has_forces": has_forces,
        "has_stress": has_stress,
        "suitable_for_training": has_energy and has_forces and config_count > 0,
    }


def analyze_xyz_files(xyz_paths: list[Path]) -> dict:
    """Aggregate analysis across multiple XYZ files."""
    if not xyz_paths:
        return {}

    elements = set()
    total_configs = 0
    total_atoms = 0
    has_energy = False
    has_forces = False
    has_stress = False

    for path in xyz_paths:
        result = analyze_xyz(path)
        elements.update(result["elements"])
        total_configs += result["n_configs"]
        total_atoms += result["n_atoms_total"]
        has_energy = has_energy or result["has_energy"]
        has_forces = has_forces or result["has_forces"]
        has_stress = has_stress or result["has_stress"]

    return {
        "elements": sorted(elements),
        "n_configs": total_configs,
        "n_atoms_total": total_atoms,
        "n_files": len(xyz_paths),
        "has_energy": has_energy,
        "has_forces": has_forces,
        "has_stress": has_stress,
        "suitable_for_training": has_energy and has_forces and total_configs > 0,
    }
