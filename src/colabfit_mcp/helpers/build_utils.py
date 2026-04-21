"""
Helper utilities for the build_dataset MCP tool.
"""

import os
import re
import tarfile
import time
from pathlib import Path


def _make_dataset_name(
    methods_contain=None,
    software_contain=None,
    formulae=None,
    elements=None,
    num_configs=1000,
) -> str:
    """Auto-generate a filesystem-safe dataset name from filter criteria."""
    parts = ["custom"]
    if methods_contain:
        parts.extend(m.replace(" ", "") for m in methods_contain[:2])
    if software_contain:
        parts.extend(s.replace(" ", "") for s in software_contain[:2])
    if formulae:
        parts.extend(formulae[:2])
    if elements:
        parts.extend(elements[:3])
    parts.append(str(num_configs))
    return re.sub(r"[^a-zA-Z0-9_]", "_", "_".join(parts))[:80]


def _poll_job(job_id: str, base_url: str, auth: tuple, timeout: int = 900) -> dict:
    """Poll /mcp/builder/status/<job_id> every 5s until complete or timeout."""
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(
            f"{base_url}/mcp/builder/status/{job_id}",
            auth=auth,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "complete":
            return data
        if status == "error":
            raise RuntimeError(data.get("error", "Unknown export error"))
        time.sleep(5)

    raise TimeoutError(f"Export job {job_id} did not complete within {timeout}s")


def _stream_and_extract(job_id: str, base_url: str, auth: tuple, output_dir: Path):
    """Download export tar.gz from /mcp/builder/download and extract to output_dir."""
    import requests

    tar_path = output_dir / "export.tar.gz"
    resp = requests.get(
        f"{base_url}/mcp/builder/download/{job_id}",
        auth=auth,
        stream=True,
        timeout=300,
    )
    resp.raise_for_status()

    with open(tar_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    with tarfile.open(str(tar_path), "r:gz") as tar:
        members = [
            m for m in tar.getmembers()
            if not os.path.isabs(m.name) and ".." not in m.name
        ]
        tar.extractall(str(output_dir), members=members)

    tar_path.unlink(missing_ok=True)


def _parquet_to_extxyz(parquet_path: Path, output_path: Path) -> tuple:
    """Convert configurations.parquet to extxyz. Returns (n_configs, elements, has_energy, has_forces, has_stress)."""
    import numpy as np
    import pyarrow.parquet as pq
    import ase.io
    from ase import Atoms

    table = pq.read_table(str(parquet_path))
    columns = set(table.schema.names)
    atoms_list = []
    all_elements: set = set()
    found_energy = False
    found_forces = False
    found_stress = False

    for i in range(len(table)):
        row = {col: table.column(col)[i].as_py() for col in columns}
        atomic_numbers = row.get("atomic_numbers") or []
        positions = row.get("positions") or []
        if not atomic_numbers or not positions:
            continue

        cell = np.array(row.get("cell") or [[0, 0, 0]] * 3, dtype=float)
        if abs(np.linalg.det(cell)) < 1e-6:
            continue
        atoms = Atoms(
            numbers=atomic_numbers,
            positions=np.array(positions, dtype=float),
            cell=cell,
            pbc=row.get("pbc") or [False, False, False],
        )
        all_elements.update(row.get("elements") or [])

        energy = row.get("energy")
        if energy is not None:
            atoms.info["energy"] = float(energy)
            found_energy = True

        forces = row.get("atomic_forces")
        if forces is not None:
            atoms.arrays["forces"] = np.array(forces, dtype=float)
            found_forces = True

        stress = row.get("cauchy_stress")
        if stress is not None:
            atoms.info["cauchy_stress"] = np.array(stress, dtype=float).flatten()
            found_stress = True

        atoms_list.append(atoms)

    ase.io.write(str(output_path), atoms_list, format="extxyz")
    return len(atoms_list), sorted(all_elements), found_energy, found_forces, found_stress


def _read_source_dataset_ids(output_dir: Path) -> list:
    """Read unique dataset IDs from configurations.parquet."""
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    parquet_path = output_dir / "configurations.parquet"
    if not parquet_path.exists():
        return []
    table = pq.read_table(str(parquet_path), columns=["dataset_id"])
    return pc.unique(table.column("dataset_id")).to_pylist()
