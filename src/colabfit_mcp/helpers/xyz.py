from pathlib import Path


def analyze_xyz(xyz_path: Path) -> dict:
    """Analyze a single XYZ/extxyz file for training suitability."""
    elements = set()
    config_count = 0
    has_energy = False
    has_forces = False
    has_stress = False
    atom_count = 0

    with open(xyz_path) as f:
        while True:
            header = f.readline()
            if not header:
                break
            line = header.strip()
            if not line:
                continue
            try:
                n_atoms = int(line)
            except ValueError:
                continue

            config_count += 1
            atom_count += n_atoms

            comment = f.readline()
            if "energy" in comment.lower():
                has_energy = True
            if "stress" in comment.lower():
                has_stress = True
            if "forces" in comment.lower():
                has_forces = True

            for _ in range(n_atoms):
                parts = f.readline().strip().split()
                if parts and parts[0].isalpha() and len(parts[0]) <= 2:
                    elements.add(parts[0])
                if len(parts) > 4:
                    has_forces = True

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
