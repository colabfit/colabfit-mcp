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
                    if val is not None and hasattr(val, "shape"):
                        logger.info(f"  batch.{attr}: shape={tuple(val.shape)} dtype={val.dtype}")
                    else:
                        logger.warning(f"  batch.{attr}: MISSING")

        def on_train_epoch_end(self, trainer, pl_module):
            from loguru import logger

            m = trainer.callback_metrics
            epoch = trainer.current_epoch
            train = float(m.get("train_loss", float("nan")))
            logger.info(f"Epoch {epoch:4d} | train_loss={train:.6f}")

        def on_validation_epoch_end(self, trainer, pl_module):
            from loguru import logger

            m = trainer.callback_metrics
            epoch = trainer.current_epoch
            val = float(m.get("val_loss", float("nan")))
            logger.info(f"Epoch {epoch:4d} | val_loss={val:.6f}")

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

            # WORKAROUND REMOVED: float32 fingerprint cast moved to kliff/kliff/trainer/lightning_trainer.py
            # setup_dataloaders() in the base class now casts fp.coords/fp.forces to float32
            # after fingerprinting when torch.get_default_dtype() == torch.float32.

            # WORKAROUND REMOVED: LightningDataset(num_workers=0) override moved to KLIFF source.
            # The base class had `if num_workers:` which treated 0 as falsy, defaulting to
            # SLURM_CPUS_PER_TASK. Fixed in KLIFF to `if num_workers is not None:` so the
            # manifest value of 0 is respected.
            logger.info("setup_dataloaders: data_module ready")

        def _get_pl_trainer(self):
            import os
            import pytorch_lightning as pl

            num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES", 1))
            strategy = self.training_manifest.get("strategy", "auto")
            accelerator = self.training_manifest.get("accelerator", "auto")
            return pl.Trainer(
                logger=[self.tb_logger, self.csv_logger],
                max_epochs=self.optimizer_manifest["epochs"],
                accelerator=accelerator,
                strategy=strategy,
                callbacks=self.callbacks,
                num_nodes=num_nodes,
                num_sanity_val_steps=0,
                enable_progress_bar=False,
            )

        def train(self):
            from loguru import logger

            logger.info("KliffTrainer.train(): calling pl_trainer.fit()")
            super().train()
            logger.info("KliffTrainer.train(): pl_trainer.fit() returned")

        def _get_callbacks(self):
            callbacks = super()._get_callbacks()
            callbacks.append(EpochProgressLogger())
            return callbacks

        # WORKAROUND REMOVED: save_kim_model override no longer needed.
        # Two fixes were made directly in kliff/kliff/trainer/lightning_trainer.py:
        #   1. deepcopy replaced with direct state_dict load (deepcopy failed on e3nn's
        #      SphericalHarmonics which stores a torch.jit.ScriptFunction as an attribute).
        #   2. weights_only=False added to torch.load for PyTorch >= 2.6 compatibility.
        # The TorchScript failure (aten::get_default_dtype) was fixed in
        # klay/klay/layers/embedding/_one_hot.py by replacing torch.get_default_dtype()
        # with a registered buffer whose .dtype is readable in TorchScript.

    return KliffTrainerWithDataset
