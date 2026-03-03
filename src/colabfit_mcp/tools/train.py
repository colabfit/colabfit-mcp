from pathlib import Path

from loguru import logger

from colabfit_mcp.config import (
    KLIFF_DEFAULTS,
    MODEL_DIR,
    container_to_host,
)
from colabfit_mcp.helpers.device import detect_device
from colabfit_mcp.helpers.kliff_trainer import get_kliff_trainer_class, run_forward_pass_check
from colabfit_mcp.helpers.training import (
    build_mace_klay_config,
    build_training_manifest,
    diagnose_failure,
    estimate_avg_num_neighbors,
    parse_kliff_metrics,
    write_mace_yaml,
)
from colabfit_mcp.helpers.kliff_utils import analyze_configs, fix_species_types
from colabfit_mcp.tools.dataset_resolver import resolve_dataset


def train_mace(
    train_file: str | None = None,
    model_name: str = "colabfit_mace",
    r_max: float = 5.0,
    max_num_epochs: int = 100,
    batch_size: int | None = None,
    device: str = None,
    elements: list[str] | None = None,
    n_layers: int = 2,
    avg_num_neighbors: float | None = None,
) -> dict:
    """Train a MACE-style KLAY model using KLIFF on XYZ data.

    When train_file is omitted, automatically discovers suitable datasets
    in the local download directory. If a matching dataset is found, it is
    loaded directly from the HuggingFace arrow cache. If no match or only a
    partial match is found, returns guidance to search and download from ColabFit.

    IMPORTANT: All file paths are inside the Docker container filesystem.
    Datasets are under /home/mcpuser/colabfit/datasets/.
    Trained models are saved under /home/mcpuser/colabfit/models/.

    Architecture notes:
        The model is a KLAY graph network with MACE-style equivariant convolutions
        built by klay.builder.build_model from a dict config. The exact config is
        written to <model_dir>/mace_model.yaml for reproducibility.

        n_layers sets the number of MACE conv layers in the *model*. The graph
        transform (RadialGraph) always uses n_layers=1 (single-hop), so all conv
        layers share the same edge_index0. This is correct for MACE — do not
        confuse model n_layers with RadialGraph n_layers.

        TorchScript export fails for this architecture because OneHotAtomEncoding
        calls torch.get_default_dtype(), which TorchScript cannot trace. The model
        is saved instead via torch.save(model, "model.pt"), which use_model loads
        with torch.load(..., weights_only=False). This is handled automatically.

    Args:
        train_file: Path to an existing extxyz file inside the container to use
            as training data. If None, auto-discovers from local downloaded datasets.
        model_name: Name for the trained model (default "colabfit_mace").
        r_max: Cutoff radius in Angstroms (default 5.0). Typical: 4-6 Å. Start
            with ~1.5x the nearest-neighbor distance. Larger = more context,
            higher cost.
        max_num_epochs: Maximum training epochs (default 100). Use 100-300 for
            scratch training; 50-100 for fine-tuning.
        batch_size: Configurations per gradient step (default 4). Reduce to 1-2
            for GPU OOM errors.
        device: "cuda", "mps", or "cpu" (auto-detected if None).
        elements: Chemical elements for dataset matching when
            train_file is not provided (e.g. ["Si", "O"]).
        n_layers: Number of MACE interaction layers (default 2). 2 is adequate
            for most systems; use 3 for complex multi-element or high-accuracy
            targets. Each additional layer increases cost significantly.
        avg_num_neighbors: Expected number of neighbors within r_max. If None,
            auto-estimated by sampling up to 20 configs from the training dataset
            using RadialGraph — no need to supply this manually. Override only
            if the estimate is known to be wrong (e.g., very small test dataset).
            Typical values: 10-40 for solid-state, 5-15 for molecular systems.

    IMPORTANT — TELL THE USER THE LOG PATH AFTER CALLING THIS TOOL:
        Training can take minutes to hours. The actual log file path is returned
        in the 'training_log' key of the result. Always report this exact path
        to the user immediately after the tool call so they can follow progress
        with `tail -f <path>` while training runs.

    Returns:
        Dict with keys:
            training_log: Host filesystem path to the training log file.
            training_log_docker: Container path to the training log file.
            model_path: Host filesystem path to the KIM model subdirectory.
            model_path_docker: Container path to the KIM model subdirectory
                (Name__MO_000000000000_000/). Pass this to use_model — NOT model_path.
            model_dir: Parent directory containing model.pt, mace_model.yaml,
                and KLIFF training logs.
            yaml_path: Path to mace_model.yaml (KLAY architecture config).
            kim_model_name, device, architecture, elements, metrics, next_step.
    """
    try:
        from kliff.dataset import Dataset
        import torch
        from omegaconf import OmegaConf
        from klay.builder import build_model
    except ImportError as e:
        return {
            "success": False,
            "error": f"Missing dependency: {e}. Install with pip install '.[full]'.",
        }

    from datetime import datetime

    model_dir = MODEL_DIR / model_name
    kim_model_name = f"{model_name}__MO_000000000000_000"
    kim_model_dir = model_dir / kim_model_name
    kim_model_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = kim_model_dir / f"training_{run_ts}.log"
    sink_id = logger.add(
        str(log_path),
        format="{time:HH:mm:ss} | {level:<8} | {message}",
        level="DEBUG",
        colorize=False,
    )
    logger.info(f"=== Training: {model_name} ===")

    if train_file is None:
        dataset_info, info = resolve_dataset(elements=elements)
        if dataset_info is None:
            logger.remove(sink_id)
            return info

        logger.info(f"Loading dataset from HuggingFace: {dataset_info['hf_id']}")
        try:
            kliff_dataset = Dataset.from_huggingface(
                dataset_info["hf_id"],
                split=dataset_info["split"],
                forces_key="atomic_forces",
                cache_dir=dataset_info["hf_cache_dir"],
            )
        except Exception as e:
            logger.remove(sink_id)
            return {"success": False, "error": f"Failed to load dataset from HuggingFace: {e}"}

        fix_species_types(kliff_dataset.configs)
        dataset_elements = elements or dataset_info["analysis"].get("elements", [])
        dataset_path = Path(dataset_info["output_dir"])

    else:
        train_path = Path(train_file)
        if not train_path.exists():
            logger.remove(sink_id)
            return {"success": False, "error": f"Training file not found: {train_file}"}

        logger.info(f"Loading dataset from file: {train_file}")
        try:
            kliff_dataset = Dataset.from_ase(
                path=str(train_path),
                energy_key="energy",
                forces_key="forces",
            )
        except Exception as e:
            logger.remove(sink_id)
            return {"success": False, "error": f"Failed to load dataset: {e}"}

        analysis = analyze_configs(kliff_dataset.configs)
        if not analysis.get("suitable_for_training"):
            logger.remove(sink_id)
            return {
                "success": False,
                "error": "Dataset not suitable for training (missing energy or forces).",
                "analysis": analysis,
            }

        dataset_elements = elements or analysis.get("elements", [])
        dataset_path = train_path.parent

    # Overwrite with actual loaded count — metadata can be stale and cause KLIFF
    # TrainerError when train_size + val_size > len(dataset)
    n_configs = len(kliff_dataset.configs)
    logger.info(f"Dataset loaded: {n_configs} configs, elements: {dataset_elements}")

    if not dataset_elements:
        logger.remove(sink_id)
        return {"success": False, "error": "Could not determine elements from dataset."}

    defaults = KLIFF_DEFAULTS
    dtype = torch.float32 if defaults["dtype"] == "float32" else torch.float64
    if dtype == torch.float32:
        import numpy as np

        for config in kliff_dataset.configs:
            if config._forces is not None:
                config._forces = config._forces.astype(np.float32)
            if config._coords is not None:
                config._coords = config._coords.astype(np.float32)
        c0 = kliff_dataset.configs[0]
        logger.info(
            f"Float32 cast: coords dtype={c0._coords.dtype}"
            f" forces dtype={c0._forces.dtype if c0._forces is not None else 'none'}"
        )

    if device is None:
        device, _ = detect_device()
    elif device not in {"cuda", "mps", "cpu"}:
        logger.remove(sink_id)
        return {
            "success": False,
            "error": f"Invalid device {device!r}. Must be 'cuda', 'mps', or 'cpu'.",
        }

    batch_size = batch_size or defaults["batch_size"]
    effective_avg_neighbors = (
        avg_num_neighbors if avg_num_neighbors is not None
        else estimate_avg_num_neighbors(kliff_dataset, r_max, dataset_elements)
    )
    logger.info(
        f"Config: {n_configs} configs | avg_neighbors={effective_avg_neighbors:.1f}"
        f" | r_max={r_max} Å | device={device} | dtype={defaults['dtype']}"
        f" | batch_size={batch_size}"
    )

    torch.set_default_dtype(dtype)

    logger.info("Building KLAY model...")
    try:
        cfg_dict = build_mace_klay_config(
            elements=dataset_elements,
            r_max=r_max,
            n_channels=defaults["num_channels"],
            lmax=defaults["lmax"],
            correlation=defaults["correlation"],
            avg_num_neighbors=effective_avg_neighbors,
            n_layers=n_layers,
        )
        yaml_path = kim_model_dir / "mace_model.yaml"
        write_mace_yaml(cfg_dict, yaml_path)
        cfg = OmegaConf.create(cfg_dict)
        model = build_model(cfg)
    except Exception as e:
        logger.remove(sink_id)
        return {"success": False, "error": f"Failed to build model: {e}"}

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model built: {n_params} parameters | n_layers={n_layers}"
        f" | lmax={defaults['lmax']} | n_channels={defaults['num_channels']}"
        f" | correlation={defaults['correlation']}"
    )

    manifest = build_training_manifest(
        dataset_path=dataset_path,
        model_name=model_name,
        model_dir=model_dir,
        elements=dataset_elements,
        r_max=r_max,
        batch_size=batch_size,
        train_size=defaults["train_size"],
        val_size=defaults["val_size"],
        max_num_epochs=max_num_epochs,
        lr=defaults["lr"],
        seed=defaults["seed"],
        device=device,
        n_configs=n_configs,
        num_workers=defaults["num_workers"],
    )

    effective_train = manifest["training"]["training_dataset"]["train_size"]
    effective_val = manifest["training"]["validation_dataset"]["val_size"]
    logger.info(
        f"Constructing trainer: train_size={effective_train} val_size={effective_val}"
        f" | batch={batch_size} | epochs={max_num_epochs}"
    )

    try:
        KliffTrainerWithDataset = get_kliff_trainer_class()
        trainer = KliffTrainerWithDataset(manifest, model=model, dataset=kliff_dataset)
    except Exception as e:
        logger.remove(sink_id)
        return {"success": False, "error": diagnose_failure(e)}

    logger.info("Trainer constructed OK")

    run_forward_pass_check(trainer, kliff_dataset, dataset_elements, r_max)

    logger.info("Calling trainer.train() — next log from inside KLIFF")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training exception: {e}")
        logger.remove(sink_id)
        return {"success": False, "error": diagnose_failure(e)}

    logger.info("Training complete.")

    try:
        _save_model_fallback(trainer, kim_model_dir, dataset_elements, r_max, kliff_dataset)
    except Exception as save_err:
        logger.error(f"Model save failed: {save_err}")

    metrics = parse_kliff_metrics(kim_model_dir)

    log_host = container_to_host(log_path)
    model_host = container_to_host(kim_model_dir)
    logger.remove(sink_id)
    return {
        "success": True,
        "training_log": log_host or str(log_path),
        "training_log_docker": str(log_path),
        "model_path": model_host or str(kim_model_dir),
        "model_path_docker": str(kim_model_dir),
        "model_dir": str(model_dir),
        "kim_model_name": kim_model_name,
        "yaml_path": str(yaml_path),
        "device": device,
        "architecture": {
            "r_max": r_max,
            "num_channels": defaults["num_channels"],
            "lmax": defaults["lmax"],
            "n_layers": n_layers,
            "correlation": defaults["correlation"],
            "avg_num_neighbors": effective_avg_neighbors,
        },
        "elements": dataset_elements,
        "metrics": metrics,
        "next_step": (
            f"KIM model saved at {kim_model_dir}. "
            "To run inference: use create_structure to build a structure file (e.g. "
            "formula='Si', crystal_structure='diamond', repeat=[2,2,2]), then pass "
            "output_file to use_model as input_file. Or pass formula+crystal_structure "
            "+repeat directly to use_model."
        ),
    }


def _write_kliff_graph_param(kim_model_dir: Path, elements: list[str], r_max: float) -> None:
    """Write kliff_graph.param so use_model can build the RadialGraph transform."""
    content = f"{len(elements)}\n{' '.join(elements)}\nGraph\n{r_max}\n1\n"
    (kim_model_dir / "kliff_graph.param").write_text(content)


def _build_trace_inputs(kliff_dataset, elements: list[str], r_max: float):
    """Return CPU tensors (species, coords, edge_index0, contributions) for torch.jit.trace."""
    import torch
    from kliff.transforms.configuration_transforms.graphs.generate_graph import RadialGraph

    transform = RadialGraph(species=elements, cutoff=r_max, n_layers=1)
    graph = transform(kliff_dataset.configs[0])
    return (
        graph.species,
        graph.coords.to(torch.float32),
        graph.edge_index0,
        graph.contributions,
    )


def _save_model_fallback(
    trainer, kim_model_dir: Path, elements: list[str], r_max: float, kliff_dataset=None
) -> None:
    import torch
    from loguru import logger

    try:
        trainer.configuration_transform.export_kim_model(str(kim_model_dir), "model_state.pt")
    except Exception:
        _write_kliff_graph_param(kim_model_dir, elements, r_max)

    model = trainer.pl_model.model.cpu()

    run_dir = Path(trainer.current.get("run_dir", ""))
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*.pth")) + sorted(ckpt_dir.glob("*.ckpt"))
        if ckpts:
            ckpt = torch.load(str(ckpts[-1]), weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                model_state = {
                    k[len("model."):]: v
                    for k, v in ckpt["state_dict"].items()
                    if k.startswith("model.")
                }
                if model_state:
                    model.load_state_dict(model_state, strict=False)

    if kliff_dataset is not None:
        try:
            class _Wrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, species, coords, edge_index0, contributions):
                    return self.m(
                        species=species,
                        coords=coords,
                        edge_index0=edge_index0,
                        contributions=contributions,
                    )

            inputs = _build_trace_inputs(kliff_dataset, elements, r_max)
            traced = torch.jit.trace(_Wrapper(model), inputs)
            torch.jit.save(traced, str(kim_model_dir / "model.pt"))
            logger.info(f"Saved traced model to {kim_model_dir / 'model.pt'}")
            return
        except Exception as trace_err:
            logger.warning(f"torch.jit.trace failed ({trace_err}), saving state dict instead")

    torch.save(model.state_dict(), str(kim_model_dir / "model_state.pt"))
    logger.info(f"Saved model_state.pt to {kim_model_dir}")
