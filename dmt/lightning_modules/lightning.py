from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch as th
from diffusers.schedulers import DDPMScheduler
from elatentlpips import ELatentLPIPS
from lightning.pytorch import LightningModule
from tqdm import tqdm
from typing_extensions import override

from dmt.models import UNetWrapper, VaeWrapper, PerceptualFeatureExtractor
from dmt.utils import (
    RankedLogger,
    compute_generator_loss,
    prediction_to_img,
    prediction_to_noise,
    set_requires_grad,
    temprngstate,
    move_tensors_to_device
)

log = RankedLogger(__name__, rank_zero_only=True)


class LitBaseModule(LightningModule):
    """The Lightning Module to train any diffusion model."""

    def __init__(
        self,
        # Models, Optimizer and Scheduler
        model_name: str,
        unet: UNetWrapper,
        vae: VaeWrapper,
        optimizer: Union[partial, List[partial]],
        scheduler: Union[partial, List[partial]],
        # Loss settings
        loss_type: str = "mse",
        perc_ckpt_path: Optional[str] = None,
        snr_gamma: Optional[float] = 1.0,
        # Noise scheduler settings
        zero_snr: bool = False,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "trailing",
        # Training settings
        gradient_accumulation: int = 1,
        gradient_checkpointing: bool = False,
        allow_tf32: bool = True,
        matmul_precision: Optional[str] = "high",
    ) -> None:
        super().__init__()

        self.unet_wrapper = unet
        self.vae_wrapper = vae
        self.partial_optimizer = optimizer
        self.partial_scheduler = scheduler

        self.loss_type = loss_type
        self.snr_gamma = snr_gamma
        self.gradient_accumulation = gradient_accumulation
        self.gradient_checkpointing = gradient_checkpointing
        self.allow_tf32 = allow_tf32
        self.matmul_precision = matmul_precision

        # Init loss
        self.lpips = None
        self.feature_extractor = None

        if loss_type == "lpips":
            # Init Latent LPIPS
            encoders = {
                "1-5": "sd15",
                "2-1": "sd21",
                "xl": "sdxl",
                "3": "sd3",
                "flux": "flux"
            }
            encoder = next((value for key, value in encoders.items() if key in model_name), None)

            self.lpips = ELatentLPIPS(encoder=encoder, augment="bg")
            set_requires_grad(self.lpips, False)
        elif loss_type == "perc":
            # Init Perceptual loss
            self.feature_extractor = (PerceptualFeatureExtractor(model_name, perc_ckpt_path)
                                      .to(self.unet_wrapper.unet.device))
            self.feature_extractor.requires_grad_(False)
            self.feature_extractor.eval()

        # Init noise scheduler
        default_sched = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
        assert not (zero_snr and (prediction_type != "v_prediction" or timestep_spacing != "trailing"))
        self.noise_scheduler = DDPMScheduler(
            default_sched.config.num_train_timesteps,
            default_sched.config.beta_start,
            default_sched.config.beta_end,
            default_sched.config.beta_schedule,
            default_sched.config.trained_betas,
            default_sched.config.variance_type,
            default_sched.config.clip_sample,
            prediction_type,
            default_sched.config.thresholding,
            default_sched.config.dynamic_thresholding_ratio,
            default_sched.config.clip_sample_range,
            default_sched.config.sample_max_value,
            timestep_spacing,
            default_sched.config.steps_offset,
            zero_snr,
        )

        # Important: Activate manual optimization
        self.automatic_optimization = False

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit and handles basic initialization."""
        if stage == "fit":
            # Gradient Checkpointing
            if self.gradient_checkpointing:
                self.vae_wrapper.vae.enable_gradient_checkpointing()
                self.unet_wrapper.unet.enable_gradient_checkpointing()

            # TF32
            if self.allow_tf32:
                th.backends.cuda.matmul.allow_tf32 = True

            # Torch Matmul precision
            if self.matmul_precision is not None:
                th.set_float32_matmul_precision(self.matmul_precision)

    @override
    def on_train_epoch_end(self) -> None:
        """Step Learning rate scheduler and log current learning rate"""
        scheduler = self.lr_schedulers()
        scheduler.step()
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

    def shared_step(self, batch: Dict[str, Any]) -> th.Tensor:
        """The forward step + loss computation"""
        ############################
        # (1) Prepare inputs
        ############################
        img = batch["pixel_values"]

        # Encode images into latent space
        latent = self.vae_wrapper.encode(img)

        # If conditioning mechanism == "concat" convert cond into latent space # TODO Move this to data loading.
        if self.unet_wrapper.cond_mechanism == "concat":
            cond = batch[self.unet_wrapper.cond_key].to(th.float32)
            if cond.max() > 1:
                cond = (cond / 127.5) - 1.0
            batch[self.unet_wrapper.cond_key] = self.vae_wrapper.encode(cond)

        # Sample noise that we'll add to the latent
        noise = th.randn_like(latent)

        # Sample a random timesteps
        timesteps = th.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (latent.shape[0],),
            device=latent.device,
        ).long()

        # Add noise to latent
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)

        ############################
        # (2) Train SD
        ############################
        # Model prediction and conversion to img (e.g. from noise or v)
        pred = self.unet_wrapper(noisy_latent, timesteps, batch)
        pred_noise = prediction_to_noise(pred, noisy_latent, timesteps, self.noise_scheduler)
        pred_img = prediction_to_img(pred, noisy_latent, timesteps, self.noise_scheduler)

        # Reconstruction loss (e.g. MSE, LPIPS, Perceptual).
        loss = compute_generator_loss(
            self,
            timesteps,
            pred,
            pred_noise,
            pred_img,
            latent,
            noise,
            batch=batch,
            loss_type=self.loss_type,
            snr_gamma=self.snr_gamma,
        )

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data
        :param batch_idx: Index of batch
        :return: A tensor of losses between model predictions and targets.
        """
        # Get optimizer and scheduler
        optimizer = self.optimizers()

        loss_train = self.shared_step(batch)
        self.manual_backward(loss_train, retain_graph=False)

        # Clip grads
        self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        # Update generator weights and set grads to None
        if (batch_idx + 1) % self.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Log loss
        self.log("train/loss", loss_train.cpu().item(), prog_bar=True, on_step=True, on_epoch=False)
        self.log("global_step", self.global_step, prog_bar=True, on_step=True, on_epoch=False)

    @th.no_grad()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> th.Tensor:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        """
        with temprngstate(batch_idx):
            loss_val = self.shared_step(batch).cpu()

        # Log loss
        self.log("val/loss", loss_val.item(), prog_bar=True, on_epoch=True)

        return loss_val

    def on_validation_epoch_end(self) -> None:
        """Manually run a sample of training images to compute a stable training loss """
        if self.global_step > 0:
            dataloader = self.trainer.datamodule.stable_train_dataloader()

            losses = []
            for i, batch in tqdm(enumerate(dataloader), desc="Stable Loss"):
                # Tensors of stable_train_dataloader are not automatically moved to cuda
                batch = move_tensors_to_device(batch, self.device)
                losses.append(self.stable_train_step(batch, i).item())

            loss = sum(losses) / len(losses)
            self.log("train/stable_loss", loss, prog_bar=True)

    @th.no_grad()
    def stable_train_step(self, batch: Dict[str, Any], batch_idx: int) -> th.Tensor:
        """Perform a single training validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        """
        with temprngstate(batch_idx):
            loss_val = self.shared_step(batch).cpu()

        return loss_val

    def get_params(self) -> List[th.nn.Parameter]:
        """Return all trainable parameters of the SD UNet (LoRA + ControlNet)."""
        params_gen = []

        for n, p in self.unet_wrapper.unet.named_parameters(recurse=True):
            # In init_unet(..) requires_grad was already set true for all trainable parameters
            # and false for anything else.
            if p.requires_grad:
                params_gen.append(p)

        return params_gen

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """Initialize optimizer and scheduler.

        :return: Optimizer and scheduler
        """
        # Init optimizers
        assert not isinstance(self.partial_optimizer, list), \
            "Normal diffusion training only supports one optimizer!"
        params = self.get_params()
        optimizer = self.partial_optimizer(params=params)
        del self.partial_optimizer

        # Init schedulers
        assert not isinstance(self.partial_scheduler, list), \
            "Normal diffusion training only supports one scheduler!"

        # Remove unsupported arguments that are there due to hydras inheritance scheme
        if (self.partial_scheduler.func == th.optim.lr_scheduler.CosineAnnealingLR or
                self.partial_scheduler.func == th.optim.lr_scheduler.LinearLR):
            self.partial_scheduler.keywords.pop("factor")
            self.partial_scheduler.keywords.pop("total_iters")
        scheduler = self.partial_scheduler(optimizer=optimizer)
        del self.partial_scheduler

        return [optimizer], [scheduler]
