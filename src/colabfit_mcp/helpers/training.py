import glob
import re
from pathlib import Path
from typing import Any


def build_mace_klay_config(
    elements: list[str],
    r_max: float,
    n_channels: int,
    lmax: int,
    correlation: int,
    avg_num_neighbors: float,
    n_layers: int = 2,
) -> dict[str, Any]:
    """Return a KLAY model config dict for a MACE-style model with n_layers conv layers."""
    num_elems = len(elements)
    layers: dict[str, Any] = {
        "element_embedding": {
            "type": "OneHotAtomEncoding",
            "config": {"num_elems": num_elems},
            "inputs": {"x": "model_inputs.species"},
        },
        "edge_features": {
            "type": "SphericalHarmonicEdgeAttrs",
            "config": {"lmax": lmax},
            "inputs": {
                "pos": "model_inputs.coords",
                "edge_index": "model_inputs.edge_index0",
            },
            "output": {0: "vectors", 1: "edge_lengths", 2: "edge_sh"},
        },
        "radial_basis": {
            "type": "RadialBasisEdgeEncoding",
            "config": {"r_max": r_max},
            "inputs": {"edge_lengths": "edge_lengths"},
        },
        "node_features_init": {
            "type": "AtomwiseLinear",
            "config": {
                "irreps_in_block": [{"l": 0, "mul": num_elems}],
                "irreps_out_block": [{"l": 0, "mul": n_channels}],
            },
            "inputs": {"h": "element_embedding"},
        },
    }

    for i in range(1, n_layers + 1):
        conv_name = f"conv{i}"
        prev = "node_features_init" if i == 1 else f"conv{i - 1}"
        input_block = (
            [{"l": 0, "mul": n_channels}]
            if i == 1
            else [{"l": 0, "mul": n_channels}, {"l": 1, "mul": n_channels}]
        )
        layers[conv_name] = {
            "type": "MACE_layer",
            "config": {
                "lmax": lmax,
                "correlation": correlation,
                "num_elements": num_elems,
                "hidden_irreps_block": [
                    {"l": 0, "mul": n_channels},
                    {"l": 1, "mul": n_channels},
                ],
                "input_block": input_block,
                "avg_num_neighbors": avg_num_neighbors,
            },
            "inputs": {
                "vectors": "vectors",
                "node_feats": prev,
                "node_attrs": "element_embedding",
                "edge_feats": "radial_basis",
                "edge_index": "model_inputs.edge_index0",
            },
        }

    last_conv = f"conv{n_layers}"
    layers["output_projection"] = {
        "type": "AtomwiseLinear",
        "config": {
            "irreps_in_block": [
                {"l": 0, "mul": n_channels},
                {"l": 1, "mul": n_channels},
            ],
            "irreps_out_block": [{"l": 0, "mul": 1}],
        },
        "inputs": {"h": last_conv},
    }
    layers["energy_sum"] = {
        "type": "KIMAPISumIndex",
        "inputs": {
            "src": "output_projection",
            "index": "model_inputs.contributions",
        },
    }

    return {
        "model_inputs": {
            "species": "Tensor (N,)",
            "coords": "Tensor (N,3)",
            "edge_index0": "Tensor (2,E)",
            "contributions": "Tensor (N,)",
        },
        "model_layers": layers,
        "model_outputs": {"energy": "energy_sum"},
    }


def write_mace_yaml(config: dict, yaml_path: Path) -> Path:
    """Write a KLAY model config dict to a YAML file."""
    import yaml

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    return yaml_path


def build_training_manifest(
    dataset_path: Path,
    model_name: str,
    model_dir: Path,
    elements: list[str],
    r_max: float,
    batch_size: int,
    train_size: int,
    val_size: int,
    max_num_epochs: int,
    lr: float,
    seed: int,
    device: str,
    n_configs: int,
    num_workers: int = 0,
    dataset_name: str | None = None,
    hf_id: str | None = None,
) -> dict[str, Any]:
    """Return a KLIFF GNNLightningTrainer manifest dict.

    dataset_path is passed as the manifest dataset path but is not loaded
    directly — the dataset is injected via KliffTrainerWithDataset.setup_dataset().
    """
    effective_train = train_size if train_size > 0 else max(1, int(n_configs * 0.9))
    effective_val = val_size if val_size > 0 else max(1, int(n_configs * 0.1))

    kim_model_name = f"{model_name}__MO_000000000000_000"
    kim_model_dir = model_dir / kim_model_name
    accelerator = "gpu" if device == "cuda" else device
    return {
        "workspace": {"name": str(kim_model_dir), "seed": seed, "resume": False},
        "model": {
            "name": model_name,
            "input_args": ["species", "coords", "edge_index0", "contributions"],
        },
        "dataset": {
            "type": "path",
            "path": str(dataset_path),
            "colabfit_dataset": {
                "database_name": "ColabFit Exchange" if hf_id else None,
                "database_url": f"https://huggingface.co/datasets/{hf_id}" if hf_id else None,
                "dataset_name": dataset_name,
            },
        },
        "transforms": {
            "configuration": {
                "name": "RadialGraph",
                "kwargs": {"cutoff": r_max, "species": elements, "n_layers": 1},
            }
        },
        "training": {
            "loss": {
                "function": "MSE",
                "weights": {"config": 1.0, "energy": 1.0, "forces": 10.0},
            },
            "optimizer": {"name": "Adam", "learning_rate": lr},
            "training_dataset": {"train_size": effective_train},
            "validation_dataset": {"val_size": effective_val},
            "batch_size": batch_size,
            "epochs": max_num_epochs,
            "num_workers": num_workers,
            "device": device,
            "accelerator": accelerator,
            "strategy": "auto",
        },
        "export": {
            "model_path": str(model_dir),
            "model_name": kim_model_name,
        },
    }


def parse_kliff_metrics(model_dir: Path) -> dict[str, Any]:
    """Read KLIFF's CSV log and return last-epoch metrics."""
    pattern = str(model_dir / "*" / "logs" / "csv_logs" / "version_*" / "metrics.csv")
    csv_files = glob.glob(pattern)
    if not csv_files:
        return {}

    csv_path = sorted(csv_files)[-1]
    metrics: dict[str, Any] = {}
    try:
        import csv

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return {}
        last = rows[-1]
        for key, val in last.items():
            if val:
                try:
                    metrics[key] = float(val)
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        from loguru import logger
        logger.warning(f"parse_kliff_metrics: failed to read {csv_path}: {e}")
    return metrics


def estimate_avg_num_neighbors(
    dataset,
    cutoff: float,
    elements: list[str],
    n_samples: int = 20,
) -> float:
    """Estimate average neighbors per atom within cutoff by sampling a KLIFF Dataset.

    Applies RadialGraph to up to n_samples configs and returns mean edges/atom.
    Falls back to 20.0 on import failure or any error.
    """
    try:
        from kliff.transforms.configuration_transforms.graphs.generate_graph import RadialGraph
    except ImportError:
        return 20.0
    try:
        configs = dataset.configs
        if not configs:
            return 20.0
        transform = RadialGraph(species=elements, cutoff=cutoff, n_layers=1)
        sample = configs[:min(n_samples, len(configs))]
        total_edges = 0
        total_atoms = 0
        for config in sample:
            graph = transform(config)
            total_edges += graph.edge_index0.shape[1]
            total_atoms += int(graph.num_nodes)
        return total_edges / total_atoms if total_atoms > 0 else 20.0
    except Exception as e:
        from loguru import logger
        logger.warning(f"estimate_avg_num_neighbors: falling back to 20.0: {e}")
        return 20.0


def diagnose_failure(exc: Exception) -> str:
    msg = str(exc).lower()
    if "cuda out of memory" in msg or "out of memory" in msg:
        return "GPU out of memory. Try reducing batch_size or using device='cpu'."
    if "nan" in msg and "loss" in msg:
        return "Training diverged (NaN loss). Try reducing lr or increasing batch_size."
    module_match = re.search(r"no module named '(\S+)'", msg)
    if module_match:
        return f"Missing dependency: {module_match.group(1)}. Install with pip install '.[full]'."
    return f"Training failed: {exc}"
