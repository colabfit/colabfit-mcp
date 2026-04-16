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
from colabfit_mcp.tools.dataset_resolver import resolve_dataset, resolve_dataset_by_name


def train_mace(
    train_file: str | None = None,
    model_name: str | None = None,
    dataset_name: str | None = None,
    r_max: float = 5.0,
    max_num_epochs: int = 100,
    batch_size: int | None = None,
    device: str = None,
    elements: list[str] | None = None,
    n_layers: int = 2,
    avg_num_neighbors: float | None = None,
) -> dict:
    """Train a MACE-style KLAY model using KLIFF on XYZ data.

    IMPORTANT: All file paths are inside the Docker container filesystem.
    Datasets are under /home/mcpuser/colabfit/datasets/.
    Trained models are saved under /home/mcpuser/colabfit/models/.

    ## DATASET SELECTION — use exactly one of: dataset_name, train_file, or elements

    Priority order: train_file > dataset_name > elements/auto-discovery.

    dataset_name (PREFERRED when you have a specific dataset):
        Exact folder name of a locally downloaded dataset, as returned by
        download_dataset (e.g. dataset_name="mlearn_Ni_test"). Performs a
        direct lookup — no scoring, no ranking, no risk of selecting a
        different dataset. ALWAYS use this when you just called download_dataset
        or when you know which local dataset you want.

    train_file (for local extxyz files):
        Container path to an extxyz file (e.g. from build_dataset). Use for
        datasets not downloaded via download_dataset. Takes precedence over
        dataset_name if both are provided.

    elements (for auto-discovery only):
        ONLY used when NEITHER dataset_name NOR train_file is provided.
        Filters auto-discovery to local datasets containing these elements.
        May select a different dataset than intended if multiple local datasets
        contain the same elements — use dataset_name to avoid ambiguity.

    If none of the three are given, auto-discovers any suitable local dataset.

    ## Architecture notes

    The model is a KLAY graph network with MACE-style equivariant convolutions
    built by klay.builder.build_model from a dict config. The exact config is
    written to <model_dir>/mace_model.yaml for reproducibility.

    n_layers sets the number of MACE conv layers in the *model*. The graph
    transform (RadialGraph) always uses n_layers=1 (single-hop), so all conv
    layers share the same edge_index0. This is correct for MACE — do not
    confuse model n_layers with RadialGraph n_layers.

    The model is exported via KLIFF's save_kim_model(), which uses
    torch.jit.script with an e3nn.util.jit fallback to handle e3nn layers
    (e.g. OneHotAtomEncoding). The result is a portable TorchScript model.pt
    compatible with the KIM TorchML driver.

    Args:
        train_file: Container path to an extxyz training file. Takes precedence
            over dataset_name. Use for custom datasets not from download_dataset.
        dataset_name: Exact local folder name returned by download_dataset
            (e.g. "mlearn_Ni_test"). Direct lookup — bypasses all discovery.
            Use this whenever you have a specific downloaded dataset to train on.
        model_name: Name for the trained model. If None, auto-generated from
            dataset name or elements.
        r_max: Cutoff radius in Angstroms (default 5.0). Typical: 4-6 Å.
        max_num_epochs: Maximum training epochs (default 100).
        batch_size: Configurations per gradient step (default 4).
        device: "cuda", "mps", or "cpu" (auto-detected if None).
        elements: Element filter for auto-discovery when dataset_name and
            train_file are both None (e.g. ["Si", "O"]).
        n_layers: Number of MACE interaction layers (default 2).
        avg_num_neighbors: Expected neighbors within r_max. Auto-estimated if None.

    IMPORTANT — TELL THE USER THE LOG PATH AFTER CALLING THIS TOOL:
        Training can take minutes to hours. Always report the 'training_log'
        path from the result so the user can follow progress with
        `tail -f <path>` while training runs.

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

    from colabfit_mcp.helpers.naming import (
        make_timestamp,
        make_model_stem,
        model_dir_name,
        kim_model_dir_name,
        training_log_name,
    )

    if train_file is not None:
        dataset_info = None
        dataset_safe_name = None
    elif dataset_name is not None:
        dataset_info, info = resolve_dataset_by_name(dataset_name)
        if dataset_info is None:
            if info.get("success") and info.get("train_file"):
                train_file = info["train_file"]
                dataset_safe_name = None
            else:
                return info
        else:
            dataset_safe_name = dataset_info["safe_name"]
    else:
        dataset_info, info = resolve_dataset(elements=elements)
        if dataset_info is None:
            return info
        dataset_safe_name = dataset_info["safe_name"]

    run_ts = make_timestamp()
    stem = make_model_stem(model_name, dataset_safe_name, elements)
    model_dir_basename = model_dir_name(stem, run_ts)
    model_dir = MODEL_DIR / model_dir_basename
    kim_model_name = kim_model_dir_name(model_dir_basename)
    kim_model_dir = model_dir / kim_model_name
    kim_model_dir.mkdir(parents=True, exist_ok=True)

    log_path = kim_model_dir / training_log_name(run_ts)
    sink_id = logger.add(
        str(log_path),
        format="{time:HH:mm:ss} | {level:<8} | {message}",
        level="DEBUG",
        colorize=False,
    )
    logger.info(f"=== Training: {model_dir_basename} ===")

    if train_file is None:
        logger.info(
            f"Loading dataset from local cache"
            f" (source: ColabFit/HuggingFace — {dataset_info['hf_id']})"
        )
        try:
            kliff_dataset = Dataset.from_huggingface(
                dataset_info["hf_id"],
                split=dataset_info["split"],
                forces_key="atomic_forces",
                cache_dir=dataset_info["hf_cache_dir"],
            )
        except Exception as e:
            logger.remove(sink_id)
            return {"success": False, "error": f"Failed to load dataset from local cache: {e}"}

        fix_species_types(kliff_dataset.configs)
        dataset_elements = dataset_info["analysis"].get("elements", [])
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

        dataset_elements = analysis.get("elements", [])
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

    _hf_id = dataset_info["hf_id"] if train_file is None else None
    _dataset_name = dataset_info["safe_name"] if train_file is None else None
    manifest = build_training_manifest(
        dataset_path=dataset_path,
        model_name=model_dir_basename,
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
        dataset_name=_dataset_name,
        hf_id=_hf_id,
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
    except RuntimeError as e:
        logger.error(f"Training exception: {e}")
        logger.remove(sink_id)
        if "device-side assert" in str(e) or "CUDA error" in str(e):
            return {
                "success": False,
                "cuda_context_poisoned": True,
                "error": (
                    "CUDA device-side assert triggered — the CUDA context is now corrupted. "
                    "All further GPU operations in this session will fail. "
                    "Restart the MCP server (restart Claude Code) to recover."
                ),
                "next_step": "Restart the MCP server by restarting Claude Code.",
            }
        return {"success": False, "error": diagnose_failure(e)}
    except Exception as e:
        logger.error(f"Training exception: {e}")
        logger.remove(sink_id)
        return {"success": False, "error": diagnose_failure(e)}

    logger.info("Training complete.")

    try:
        trainer.save_kim_model()
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
            "INFERENCE: call use_model with formula+crystal_structure+repeat, or "
            "create_structure first then pass output_file to use_model as input_file. "
            "TEST DRIVERS: ALWAYS call list_test_drivers() first — it returns "
            "crystal_structure_info (exact formula requirements per structure type), "
            "crystal_structure_examples (verified material→structure+lattice_constant mappings), "
            "and per-driver workflow guidance. Key rules: "
            "(1) crystal_structure must be an ASE bulk() name, not a mineral name — "
            "e.g. TiO2 uses crystal_structure='fluorite' lattice_constant=4.59, NOT 'tetragonal' or 'rutile'; "
            "(2) lattice_constant is ALWAYS required for compound formulas "
            "(rocksalt/zincblende/wurtzite/cesiumchloride/fluorite); "
            "(3) ClusterEnergyAndForces auto-converts crystal structures to non-periodic clusters "
            "when crystal_structure is passed — do not call create_structure first."
        ),
    }
