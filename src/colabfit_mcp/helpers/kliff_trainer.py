def run_forward_pass_check(trainer, dataset, elements: list[str], r_max: float) -> None:
    """Run one forward pass before training to validate model + graph transform.

    Applies RadialGraph to the first config, feeds it through the model, and logs
    the output shape/value. Any exception is caught and logged without aborting training.
    """
    from loguru import logger

    logger.info("Running dry-run forward pass check...")
    try:
        from kliff.transforms.configuration_transforms.graphs.generate_graph import RadialGraph
        import torch

        config = dataset.configs[0]
        transform = RadialGraph(species=elements, cutoff=r_max, n_layers=1)
        graph = transform(config)

        pl_model = getattr(trainer, "pl_model", None)
        if pl_model is None:
            logger.warning("Dry-run skipped: trainer.pl_model not yet initialized")
            return
        model = pl_model.model

        inputs = {
            "species": torch.as_tensor(graph.species, dtype=torch.long),
            "coords": torch.as_tensor(graph.coords, dtype=torch.float32),
            "edge_index0": torch.as_tensor(graph.edge_index0, dtype=torch.long),
            "contributions": torch.as_tensor(graph.contributions, dtype=torch.long),
        }
        model.eval()
        with torch.no_grad():
            output = model(**inputs)
        logger.info(
            f"Dry-run OK: output shape={tuple(output.shape)}"
            f" dtype={output.dtype} mean={float(output.mean()):.4f}"
        )
    except Exception as e:
        logger.warning(f"Dry-run forward pass failed (training may still work): {e}")


def _apply_klay_debug_patches() -> None:
    """Monkey-patch KLAY builder with loguru tracing when COLABFIT_DEBUG=1."""
    import os

    if os.getenv("COLABFIT_DEBUG") != "1":
        return
    from loguru import logger

    try:
        from klay.builder import fx_builder

        _orig = fx_builder.build_fx_model

        def _patched(cfg, **kw):
            try:
                from omegaconf import OmegaConf
                d = OmegaConf.to_container(cfg, resolve=True)
            except Exception:
                d = {}
            logger.debug(f"[KLAY build] model_inputs: {list(d.get('model_inputs', {}).keys())}")
            logger.debug(f"[KLAY build] layers: {list(d.get('model_layers', {}).keys())}")
            result = _orig(cfg, **kw)
            logger.debug("[KLAY build] build_fx_model returned")
            return result

        fx_builder.build_fx_model = _patched
        logger.info("[DEBUG] KLAY monkey-patch applied (COLABFIT_DEBUG=1)")
    except Exception as e:
        logger.warning(f"[DEBUG] KLAY monkey-patch failed: {e}")


def get_kliff_trainer_class():
    """Return a GNNLightningTrainer subclass that accepts a pre-loaded KLIFF Dataset.

    The base GNNLightningTrainer has no dataset= kwarg — its __init__ calls
    setup_dataset() which loads from the manifest path. This subclass overrides
    setup_dataset() so that a pre-loaded Dataset is used directly, bypassing
    manifest-based file loading entirely.
    """
    from kliff.trainer.lightning_trainer import GNNLightningTrainer
    from pytorch_lightning.callbacks import Callback

    _apply_klay_debug_patches()

    class EpochProgressLogger(Callback):
        """Log per-epoch train/val loss and first-batch tensor shapes via loguru."""

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if batch_idx == 0 and trainer.current_epoch == 0:
                from loguru import logger

                for attr in ["species", "coords", "energy", "forces", "images", "contributions", "edge_index0"]:
                    val = getattr(batch, attr, None)
                    if val is not None:
                        logger.info(f"  batch.{attr}: shape={tuple(val.shape)} dtype={val.dtype}")
                    else:
                        logger.warning(f"  batch.{attr}: MISSING")

        def on_validation_epoch_end(self, trainer, pl_module):
            from loguru import logger

            m = trainer.callback_metrics
            epoch = trainer.current_epoch
            train = float(m.get("train_loss", float("nan")))
            val = float(m.get("val_loss", float("nan")))
            logger.info(f"Epoch {epoch:4d} | train_loss={train:.6f} | val_loss={val:.6f}")

    class KliffTrainerWithDataset(GNNLightningTrainer):
        def __init__(self, manifest, model=None, dataset=None):
            from loguru import logger

            logger.info("KliffTrainerWithDataset.__init__: start")
            self._injected_dataset = dataset
            logger.info("KliffTrainerWithDataset.__init__: calling super().__init__ (KLIFF init)")
            super().__init__(manifest, model)
            logger.info("KliffTrainerWithDataset.__init__: super().__init__ complete")

        def setup_dataset(self):
            from loguru import logger

            if self._injected_dataset is not None:
                logger.info(
                    f"setup_dataset: injecting pre-loaded dataset"
                    f" ({len(self._injected_dataset.configs)} configs)"
                )
                self.dataset = self._injected_dataset
            else:
                logger.info("setup_dataset: loading from manifest path")
                super().setup_dataset()

        def setup_dataloaders(self):
            from loguru import logger

            n_total = (
                len(self.dataset.configs)
                if hasattr(self, "dataset") and self.dataset is not None
                else "?"
            )
            logger.info(f"setup_dataloaders: fingerprinting {n_total} configs (blocking)...")
            super().setup_dataloaders()

            n_train = len(self.train_dataset) if hasattr(self, "train_dataset") and self.train_dataset is not None else "?"
            n_val = len(self.val_dataset) if hasattr(self, "val_dataset") and self.val_dataset is not None else "?"
            logger.info(f"setup_dataloaders: split complete → {n_train} train + {n_val} val")

            try:
                from torch_geometric.data.lightning_datamodule import LightningDataset
            except ImportError:
                from torch_geometric.data.lightning import LightningDataset
            self.data_module = LightningDataset(
                self.train_dataset,
                self.val_dataset,
                batch_size=self.optimizer_manifest["batch_size"],
                num_workers=0,
            )
            logger.info("setup_dataloaders: data_module ready (num_workers=0)")

        def train(self):
            from loguru import logger

            logger.info("KliffTrainer.train(): calling pl_trainer.fit()")
            super().train()
            logger.info("KliffTrainer.train(): pl_trainer.fit() returned")

        def _get_callbacks(self):
            callbacks = super()._get_callbacks()
            callbacks.append(EpochProgressLogger())
            return callbacks

    return KliffTrainerWithDataset
