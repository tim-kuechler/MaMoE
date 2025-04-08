from typing import List

import hydra
from lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import rootutils

from mamoe.utils import (
    RankedLogger,
    apply_extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> None:
    """Trains the model.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        log.info(f"Setting seed to <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating pytorch lightning callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    with trainer.init_module():
        model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Logging hyperparameters
    log_object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(log_object_dict)

    # Train via lightning trainer
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for training basic diffusion models.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. disable python warnings, print cfg tree, etc.)
    apply_extras(cfg)

    # Run training code
    train(cfg)


if __name__ == "__main__":
    main()
