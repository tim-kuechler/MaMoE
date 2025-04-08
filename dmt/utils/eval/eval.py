import csv
import os
import shutil
from typing import Optional

import torch as th
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger

from dmt.utils.eval.cmmd import compute_cmmd
from dmt.utils.eval.fid import compute_fid, compute_kid
from dmt.utils.eval.sample import sample_images


def _save_metric(
        metric: str,
        metric_value: float,
        metrics_dir: str,
        mode: str,
        step: Optional[int] = None
) -> None:
    """Write a calculated metric to a csv file.

    :param metric: The name of the metric.
    :param metric_value: The calculated metric.
    :param metrics_dir: The folder where the output file should be saved.
    :param mode: A "mode" variable that is part of the filename and can be set to any string.
    :param step: The training step at which the metric was calculated. Use e.g. 0 for test metrics.
    """
    metric_file = os.path.join(metrics_dir, f"{metric}_{mode}.csv")
    file_exists = os.path.exists(metric_file) and os.path.getsize(metric_file) > 0
    with open(metric_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["step", metric] if step is not None else [metric])

        if not file_exists:
            writer.writeheader()

        writer.writerow({"step": step, metric: metric_value} if step is not None else {metric: metric_value})


@th.inference_mode()
def calculate_metrics(
    lit_module: LightningModule,
    trainer: Trainer,
    eval_fid: bool = False,
    eval_kid: bool = False,
    eval_cmmd: bool = True,
    split: str = "test",
    step: Optional[int] = None,
    cond: bool = False,
    uncond: bool = False,
    cfg: bool = False,
    delete_samples: bool = True,
    num_images: Optional[int] = None,
    metric_batch_size: int = 32,
    log_wandb: bool = True,
    sampling_batch_size: Optional[int] = None,
) -> None:
    """Samples validation or test images and calculates FID, KID and CMMD metrics. Metrics will be saved to disk
    and to wandb if initialized.

    :param lit_module: PyTorch Lightning Module.
    :param trainer: PyTorch Lightning Trainer.
    :param eval_cmmd: If true, evaluate CMMD.
    :param eval_fid: If true, evaluate FID.
    :param eval_kid: If true, evaluate KID.
    :param cond: Sample images conditionally (i.e. with prompt and without CFG).
    :param uncond: Sample images unconditionally (i.e. without prompt / with "" prompt).
    :param cfg: Sample images with Classifier-free guidance (with prompts).
    :param delete_samples: If True, delete the generated samples after score computation.
    :param num_images: The number of validation images to sample.
    :param metric_batch_size: The batch size used during metric calculation.
    :param log_wandb: If true, log to wandb (if initialized).
    :param sampling_batch_size: The batch size used during test image sampling.
    """
    if not cond and not uncond and not cfg:
        return

    reference_imgs_dir = os.path.join(trainer.datamodule.dataset_dir, "images", split)
    metrics_dir = os.path.join(trainer.default_root_dir, "eval", split, "metrics")
    samples_dir = os.path.join(metrics_dir, "samples", f"step_{str(step)}") if step is not None else os.path.join(metrics_dir, "samples")

    if num_images is None:
        num_images = len(os.listdir(reference_imgs_dir))

    # Get modes to compute fid for
    modes = []
    if cond:
        modes.append("cond")
    if uncond:
        modes.append("uncond")
    if cfg:
        modes.append("cfg")

    # Generate images
    sample_images(
        lit_module,
        trainer,
        samples_dir,
        save_wandb=False,
        save_local=True,
        num_steps=50,
        num_batches=num_images // trainer.datamodule.batch_size,
        batch_size=trainer.datamodule.batch_size if sampling_batch_size is None else sampling_batch_size,
        modes=modes,
        split=split,
        shuffle=False,
        seed=42,
    )

    # Compute metrics
    for mode in modes:
        if eval_fid:
            fid_value = compute_fid(
                reference_imgs_dir,
                os.path.join(samples_dir, mode),
                mode="clean",
                batch_size=metric_batch_size,
                device=lit_module.device
            )

            if log_wandb and isinstance(trainer.logger, WandbLogger):
                trainer.logger.experiment.log({f"{split}/fid_{mode}": fid_value})

            _save_metric("fid", fid_value, metrics_dir, mode, step)

        if eval_kid:
            kid_value = compute_kid(
                reference_imgs_dir,
                os.path.join(samples_dir, mode),
                mode="clean",
                batch_size=metric_batch_size,
                device=lit_module.device
            )

            if log_wandb and isinstance(trainer.logger, WandbLogger):
                trainer.logger.experiment.log({f"{split}/kid_{mode}": kid_value})

            _save_metric("kid", kid_value, metrics_dir, mode, step)

        if eval_cmmd:
            cmmd_value = compute_cmmd(
                reference_imgs_dir,
                os.path.join(samples_dir, mode),
                metric_batch_size,
                num_images,
            ).tolist()

            if log_wandb and isinstance(trainer.logger, WandbLogger):
                trainer.logger.experiment.log({f"{split}/cmmd_{mode}": cmmd_value})

            _save_metric("cmmd", cmmd_value, metrics_dir, mode, step)

    if delete_samples:
        shutil.rmtree(os.path.join(metrics_dir, "samples"))
