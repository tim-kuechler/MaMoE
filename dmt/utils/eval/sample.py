import copy
import os.path
from typing import List, Optional

import torch as th
import wandb
from diffusers.schedulers import DDIMScheduler
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms.v2.functional import resize, InterpolationMode
from torchvision.utils import save_image
from tqdm import tqdm

from dmt.utils import temprngstate, move_tensors_to_device, change_tensors_to_dtype


@th.inference_mode()
def sample_images(
        lit_module: LightningModule,
        trainer: Trainer,
        sample_dir: Optional[str] = None,
        save_wandb: bool = True,
        save_local: bool = False,
        num_steps: int = 50,
        num_batches: int = 1,
        batch_size: int = 1,
        modes: List[str] = ["cond"],
        split: str = "val",
        shuffle: bool = False,
        seed: Optional[int] = None,
        cfg_guidance_scale: float = 4.0,
        cfg_guidance_rescale: float = 0.7,
        use_float16: bool = True,
) -> None:
    """Samples images.

    :param lit_module: PyTorch Lightning Module.
    :param trainer: PyTorch Lightning Trainer.
    :param sample_dir: Directory where the generated images should be saved.
    :param save_wandb: If true, log images to wandb (if initialized).
    :param save_local: If true, log images to disk.
    :param num_steps: Number of sampling steps.
    :param num_batches: The number of batches to sample.
    :param batch_size: The batch size to be used for sampling.
    :param modes: A list of modes to sample for (can contain "cond", "uncond", "cfg")
    :param split: The dataloader to use for sampling. One of ("train", "val", "test")
    :param shuffle: If true, the dataloader is shuffled.
    :param seed: A seed for sampling the noise during sampling (for deterministic or comparable samples).
    :param cfg_guidance_scale: The guidance scale for Classifier-free guidance (CFG).
    :param cfg_guidance_rescale: The rescale factor for CFG. Should only be used when ZeroSNR is activated.
    :param use_float16: If true, sample with float16 instead of float32.
    """
    with temprngstate(seed):
        # Get dataloader
        dataloaders = {
            "train": lambda: trainer.datamodule.train_dataloader(batch_size=batch_size, shuffle=shuffle),
            "val": lambda: trainer.datamodule.val_dataloader(batch_size=batch_size, full_dataset=True, shuffle=shuffle),
            "test": lambda: trainer.datamodule.test_dataloader(batch_size=batch_size, shuffle=shuffle)
        }
        dataloader = dataloaders[split]()
        assert dataloader, f"Dataloader for split {split} is None."

        # Prepare scheduler and model
        device = lit_module.device
        config = lit_module.noise_scheduler.config
        # Unfortunately DDIMScheduler.from_config(config) seems not to work correctly
        scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            prediction_type=config.prediction_type,
            thresholding=config.thresholding,
            dynamic_thresholding_ratio=config.dynamic_thresholding_ratio,
            clip_sample_range=config.clip_sample_range,
            sample_max_value=config.sample_max_value,
            timestep_spacing=config.timestep_spacing,
            rescale_betas_zero_snr=config.rescale_betas_zero_snr
        )
        model = lit_module.unet_wrapper if not use_float16 else copy.deepcopy(lit_module.unet_wrapper).to(th.float16)
        vae = lit_module.vae_wrapper if not use_float16 else copy.deepcopy(lit_module.vae_wrapper).to(th.float16)
        model.eval()

        # Prepare timesteps
        scheduler.set_timesteps(num_steps, device=device)
        timesteps = scheduler.timesteps

        pbar = tqdm(total=len(modes) * num_batches * batch_size, desc="Sampling images...")
        for mode in modes:
            assert mode in ("uncond", "cond", "cfg")

            i = 0
            sampled_images = []
            cond_images = []
            for batch in dataloader:
                if i == num_batches:
                    break

                batch = move_tensors_to_device(batch, device)
                if use_float16:
                    batch = change_tensors_to_dtype(batch, th.float16)

                # Prepare latents
                shape = (
                    batch_size,
                    4,
                    int(batch["pixel_values"].shape[2]) // 8,
                    int(batch["pixel_values"].shape[3]) // 8,
                )
                latents = th.randn(shape, device=device, dtype=model.unet.dtype)

                # Denoising loop
                samples = latents
                for t in timesteps:
                    if  mode == "cond" or mode == "cfg":
                        pred_cond = model(samples, t, batch, cn_dropout=0.0, txt_dropout=0.0)
                    if  mode == "uncond" or mode == "cfg":
                        pred_uncond = model(samples, t, copy.deepcopy(batch), cn_dropout=0.0, txt_dropout=1.0)

                    if mode == "cfg":
                        pred = pred_uncond + cfg_guidance_scale * (pred_cond - pred_uncond)
                        pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=cfg_guidance_rescale if config.rescale_betas_zero_snr else 0.0)
                    else:
                        pred = pred_cond if mode == "cond" else pred_uncond

                    samples = scheduler.step(pred, t, samples, return_dict=False)[0]

                # Decoding
                images = vae.decode(samples)

                # Postprocessing
                images = (images / 2 + 0.5).clamp(0, 1)

                sampled_images.append(images.cpu())

                # Get condition images
                if "uncond" not in mode:
                    cond_imgs = (batch["pixel_values"] / 2 + 0.5).clamp(0, 1)
                    cond_h, cond_w = cond_imgs.shape[-2], cond_imgs.shape[-1]
                    if lit_module.unet_wrapper.cond_key != "pixel_values":
                        cond_img = resize(batch[lit_module.unet_wrapper.cond_key] / 255.0,
                                                [cond_imgs.shape[-2], cond_imgs.shape[-1]],
                                                InterpolationMode.NEAREST_EXACT)
                        cond_imgs = th.cat((cond_img, cond_imgs), dim=-1)
                    if "panoptic_img" in batch:
                        panoptic_image = resize(batch["panoptic_img"] / 255.0,
                                                [cond_h, cond_w], InterpolationMode.NEAREST_EXACT)
                        cond_imgs = th.cat((panoptic_image, cond_imgs), dim=-1)
                    if "moe_binary_mask" in batch:
                        moe_mask = resize(batch["moe_binary_mask"], [cond_h, cond_w], InterpolationMode.NEAREST_EXACT)
                        moe_mask = moe_mask.unsqueeze(dim=1).repeat(1, 3, 1, 1)
                        cond_imgs = th.cat((moe_mask, cond_imgs), dim=-1)
                    cond_images.append(cond_imgs.float().cpu())

                i += 1
                pbar.update(batch_size)

            samples = th.cat(sampled_images, dim=0)
            conds = th.cat(cond_images, dim=0) if "uncond" not in mode else None

            if save_wandb and isinstance(trainer.logger, WandbLogger):
                wandb_images = th.cat((conds, samples), dim=-1) if conds is not None else samples
                wandb_images = [wandb.Image(wandb_images[idx], caption=f"idx={idx}") for idx in range(wandb_images.shape[0])]
                trainer.logger.experiment.log({f"{split}/samples_{mode}": wandb_images})

            if save_local:
                assert sample_dir is not None
                os.makedirs(os.path.join(sample_dir, mode), exist_ok=True)
                for i in range(samples.shape[0]):
                    save_image(samples[i], os.path.join(sample_dir, mode, f"img_{mode}_{i}.png"))

        model.train()
        if use_float16:
            del model
            th.cuda.empty_cache()


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    if guidance_rescale > 0.0:
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg

    return noise_cfg