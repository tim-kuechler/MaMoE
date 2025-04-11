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
        optimizer: partial,
        scheduler: partial,
        # Loss settings
        loss_type: str = "mse",
        perc_ckpt_path: Optional[str] = None,
        snr_gamma: Optional[float] = None,
        # Noise scheduler settings
        zero_snr: bool = False,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "trailing",
        # Training settings
        gradient_accumulation: int = 1,
        gradient_checkpointing: bool = False,
        allow_tf32: bool = True,
        matmul_precision: Optional[str] = "high",
        continue_epoch: Optional[int] = None,
        continue_step: Optional[int] = None,
    ) -> None:
        """Init the Training Loop

        :param model_name: The Huggingface model repository of the model that was loaded.
        :param unet: A UnetWrapper around an initialized model (is initialized by hydra).
        :param vae: A VaeWrapper around an initialized VAE (is initialized by hydra).
        :param optimizer: A partial function that initializes the optimizer and only misses trainable parameters.
        :param scheduler: A partial function that initializes the LR scheduler and only misses trainable parameters.
        :param loss_type: Training loss. Either "mse", "perc" or "lpips".
        :param perc_ckpt_path: Ckpt to load for the perceptual feature descriptor (if loss is "perc").
        :param snr_gamma: Loss weighting factor for MinSNR.
        :param zero_snr: If true, use ZeroSNR fix (in this case prediction_type must be "v_prediction").
        :param prediction_type: The prediction type of the model, either "epsilon" or "v_prediction"
            (maybe "sample" also works).
        :param timestep_spacing: The timestep spacing during sampling. Refer to ZeroSNR paper for explanation and modes.
        :param gradient_accumulation: How many training steps are accumulated before a backpropagation.
        :param gradient_checkpointing: If true, activate gradient checkpointing to trade of speed for VRAM.
        :param allow_tf32: If true, train with tensorfloat32.
        :param matmul_precision: Define PyTorch's matrix multiplication precision.
        :param continue_epoch: Optional. If set start with this epoch (e.g. to continue training).
        :param continue_step: Optional. If set start with this training step. Also, epoch and step can be set both.
        """
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
        self.continue_epoch = continue_epoch
        self.continue_step = continue_step

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

    ################ Code to set start epoch and step to continue training ################
    @override
    def on_train_start(self) -> None:
        if self.continue_epoch is not None or self.continue_step is not None:
            if self.continue_epoch is None:
                self.continue_epoch = 0

            # Set epoch
            if self.continue_epoch > 0:
                self.trainer.fit_loop.epoch_progress.current.completed = self.continue_epoch
                self.trainer.fit_loop.epoch_progress.current.processed = self.continue_epoch
                assert self.current_epoch == self.continue_epoch, f"{self.current_epoch} != {self.continue_epoch}"

            # Set batch id
            if self.continue_step is not None:
                total_batch_idx = self.continue_step
            else:
                total_batch_idx = self.current_epoch * len(self.trainer.train_dataloader)
            self.trainer.fit_loop.epoch_loop.batch_progress.total.ready = total_batch_idx + 1
            self.trainer.fit_loop.epoch_loop.batch_progress.total.completed = total_batch_idx
            assert self.trainer.fit_loop.epoch_loop.total_batch_idx + 1 == total_batch_idx + 1, \
                f"{self.trainer.fit_loop.epoch_loop.total_batch_idx + 1} != {total_batch_idx + 1}"

            # Set global step
            global_step = total_batch_idx
            self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed = global_step
            self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed = global_step
            assert self.global_step == global_step, f"{self.global_step} != {global_step}"

            # Tick LR Scheduler for every epoch
            for i in range(self.current_epoch):
                self.lr_schedulers().step()

    ################################################################################

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
        params = self.get_params()
        optimizer = self.partial_optimizer(params=params)
        del self.partial_optimizer

        # Init schedulers
        # Remove unsupported arguments that are there due to hydras inheritance scheme
        if (self.partial_scheduler.func == th.optim.lr_scheduler.CosineAnnealingLR or
                self.partial_scheduler.func == th.optim.lr_scheduler.LinearLR):
            self.partial_scheduler.keywords.pop("factor")
            self.partial_scheduler.keywords.pop("total_iters")
        scheduler = self.partial_scheduler(optimizer=optimizer)
        del self.partial_scheduler

        return [optimizer], [scheduler]
