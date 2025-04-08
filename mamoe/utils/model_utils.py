import os
import random
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union, Tuple, Set

import numpy as np
import torch as th
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import DDPMScheduler
from diffusers.training_utils import compute_snr
from lightning.pytorch import LightningModule
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)


def set_requires_grad(
    module: th.nn.Module, requires_grad: bool, parameter_list: Optional[List] = None
) -> List[str]:
    """Sets requires_grad to the requested value for all parameter in parameter_list or - if paramater_list
    is None - to all parameters in the module.

    :param module: Torch module to set grad parameters for.
    :param requires_grad: True or False.
    :param parameter_list: List of parameters to change. If None changes all parameters.
    :return: List of changed parameters (name as str).
    """
    changed_params = []

    for name, param in module.named_parameters():
        # Change parameter only if the requested requires_grad is different from current state
        # and if parameter_list exists the parameter must be in the list.
        if param.requires_grad != requires_grad and not (
            parameter_list is not None and name not in parameter_list
        ):
            param.requires_grad = requires_grad
            changed_params.append(name)

    return changed_params


def prediction_to_img(
        model_output: th.Tensor,
        noisy_latent: th.Tensor,
        timesteps: th.Tensor,
        noise_scheduler: DDPMScheduler,
) -> th.Tensor:
    """Given the prediction output of the UNet, calculates the predicted image.

    :param model_output: The model's prediction.
    :param noisy_latent: The noisy image that was passed as an input to the model (B, 4, h, w).
    :param timesteps: The timesteps used to add noise to the latent image (B,).
    :param noise_scheduler: The DDPM Noise scheduler used to sample the noise.
    :return: Predicted image without noise.
    """
    # Compute alphas, betas
    alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps][:, None, None, None]
    beta_prod_t = 1 - alpha_prod_t

    # Compute original sample (x_0) from model prediction
    if noise_scheduler.config.prediction_type == "epsilon":
        pred_img = (noisy_latent - (beta_prod_t ** 0.5) * model_output) / (alpha_prod_t ** 0.5)
    elif noise_scheduler.config.prediction_type == "sample":
        pred_img = model_output
    elif noise_scheduler.config.prediction_type == "v_prediction":
        pred_img = (alpha_prod_t**0.5) * noisy_latent - (
            beta_prod_t**0.5
        ) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {noise_scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    return pred_img


def prediction_to_noise(
        model_output: th.Tensor,
        noisy_latent: th.Tensor,
        timesteps: th.Tensor,
        noise_scheduler: DDPMScheduler,
) -> th.Tensor:
    """Given the prediction output of the UNet, calculates the predicted noise.

    :param model_output: The model's prediction.
    :param noisy_latent: The noisy image that was passed as an input to the model (B, 4, h, w).
    :param timesteps: The timesteps used to add noise to the latent image (B,).
    :param noise_scheduler: The DDPM Noise scheduler used to sample the noise.
    :return: Predicted noise that was added to the latent image.
    """
    # Compute alphas, betas
    alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps][:, None, None, None]
    beta_prod_t = 1 - alpha_prod_t

    # Compute predicted noise from model prediction
    if noise_scheduler.config.prediction_type == "epsilon":
        pred_noise = model_output
    elif noise_scheduler.config.prediction_type == "sample":
        pred_noise = th.zeros_like(model_output)
    elif noise_scheduler.config.prediction_type == "v_prediction":
        pred_noise = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * noisy_latent
    else:
        raise ValueError(
            f"prediction_type given as {noise_scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    return pred_noise


def compute_generator_loss(
    pl_module: LightningModule,
    timesteps: th.Tensor,
    model_pred: Optional[th.Tensor] = None,
    model_pred_noise: Optional[th.Tensor] = None,
    model_pred_img: Optional[th.Tensor] = None,
    orig_img: Optional[th.Tensor] = None,
    orig_noise: Optional[th.Tensor] = None,
    batch: Optional[Dict[str, Any]] = None,
    loss_type: str = "mse",
    snr_gamma: Optional[float] = None,
) -> th.Tensor:
    """Computes the loss for generator or discriminator between model prediction and target. (Can
    be different types depending on if eps or v prediction).

    Supports MSE loss, LPIPS loss, Perceptual latent loss or no loss. Also supports timestep
    weighting according to Min-SNR weighting.

    :param model_pred: Model prediction.
    :param target: Target to compute loss against.
    :param noise_scheduler: The noise scheduler used to sample the timesteps.
    :param timesteps: Timesteps.
    :param loss_type: Type of loss to compute (adv, mse, lpips, perceptual or none)
    :param snr_gamma: None or snr_gamma value for Min-SNR weighting for timesteps
    :return: Loss as tensor.
    """
    if snr_gamma is not None:
        snr = compute_snr(pl_module.noise_scheduler, timesteps)
        snr_loss_weights = th.stack([snr, snr_gamma * th.ones_like(timesteps)], dim=1)
        snr_loss_weights = snr_loss_weights.min(dim=1)
        snr_loss_weights = snr_loss_weights[0]

        if pl_module.noise_scheduler.config.prediction_type == "epsilon":
            snr_loss_weights = snr_loss_weights / snr
        elif pl_module.noise_scheduler.config.prediction_type == "v_prediction":
            snr_loss_weights = snr_loss_weights / (snr + 1)
    else:
        snr_loss_weights = th.ones_like(timesteps)

    if loss_type == "mse":
        target = _get_mse_target(orig_img, orig_noise, pl_module.noise_scheduler, timesteps)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * snr_loss_weights
        loss = loss.mean()
    elif loss_type == "lpips":
        assert hasattr(pl_module, "lpips")
        loss = pl_module.lpips(model_pred_img, orig_img, normalize=False).squeeze()
        loss = (loss * snr_loss_weights).mean()
    elif loss_type == "perc":
        assert pl_module.feature_extractor is not None and batch is not None

        # Sample new timesteps
        renoising_timesteps = th.randint(
            0,
            pl_module.noise_scheduler.config.num_train_timesteps,
            (orig_img.shape[0],),
            device=orig_img.device,
        ).long()

        # Renoise predicted image with predicted noise
        x_pred_renoised = pl_module.noise_scheduler.add_noise(model_pred_img, model_pred_noise, renoising_timesteps)

        # Renoise original image with original noise
        x_orig_renoised = pl_module.noise_scheduler.add_noise(orig_img, orig_noise, renoising_timesteps)

        # Extract perceptual features
        x_pred_features = pl_module.feature_extractor(x_pred_renoised, renoising_timesteps, batch)
        x_orig_features = pl_module.feature_extractor(x_orig_renoised, renoising_timesteps, batch)

        loss = F.mse_loss(x_pred_features, x_orig_features, reduction="mean")
    elif loss_type == "none":
        return th.tensor(0).to(timesteps.device)
    else:
        raise NotImplementedError(f"Currently loss type {loss_type} is not implemented")

    return loss


def _get_mse_target(
    latent: th.Tensor, noise: th.Tensor, noise_scheduler: DDPMScheduler, timesteps: th.IntTensor
) -> th.Tensor:
    """Computes target to calculate loss against.

    Depending on if eps or v prediction returns noise or velocity.

    :param latent: The latent original (input) image (w/o noise).
    :param noise: The noise that will be added to the latent.
    :param noise_scheduler: The noise scheduler used to sample the timesteps.
    :param timesteps: Timesteps.
    :return: Target as tensor.
    """
    if noise_scheduler.config.prediction_type == "epsilon":
        return noise
    elif noise_scheduler.config.prediction_type == "sample":
        return latent
    elif noise_scheduler.config.prediction_type == "v_prediction":
        return noise_scheduler.get_velocity(latent, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


@contextmanager
def temprngstate(new_seed: Optional[int] = None) -> None:
    """Context manager that saves and restores the RNG state of PyTorch, NumPy and Python.

    :param new_seed: The temporary seed to set in the declared context
    """

    # Save RNG state
    old_torch_rng_state = th.get_rng_state()
    old_torch_cuda_rng_state = th.cuda.get_rng_state()
    old_numpy_rng_state = np.random.get_state()
    old_python_rng_state = random.getstate()

    # Set new seed
    if new_seed is not None:
        th.manual_seed(new_seed)
        th.cuda.manual_seed(new_seed)
        np.random.seed(new_seed)
        random.seed(new_seed)

    yield

    # Restore RNG state
    th.set_rng_state(old_torch_rng_state)
    th.cuda.set_rng_state(old_torch_cuda_rng_state)
    np.random.set_state(old_numpy_rng_state)
    random.setstate(old_python_rng_state)


def encode_prompt(model_name: str, prompt: str, squeeze: bool = True) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
    """Encodes a prompt for a model.

    :param model_name: Huggingface model name to encode prompt for.
    :param prompt: The prompt to encode.
    :param squeeze: If true remove the first (batch) dimension of the encoded prompt (which is always 1).
    :return: Prompt encodings as tensor.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if "xl" in model_name:
        tokenizer_one = AutoTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer",
            use_fast=True,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer_2",
            use_fast=True,
        )

        text_encoder_one = CLIPTextModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
        ).cuda()
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            model_name,
            subfolder="text_encoder_2",
        ).cuda()
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)


        def enc_prompt(prompt: str):
            prompt_embeds_list = []

            for tokenizer, text_encoder in zip(
                    [tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two]
            ):
                tokens = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.cuda()

                prompt_embeds = text_encoder(tokens, output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = th.concat(prompt_embeds_list, dim=-1).cpu()
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1).cpu()

            # Clean VRAM
            del tokenizer_one, tokenizer_two
            del text_encoder_one, text_encoder_two
            th.cuda.empty_cache()

            if squeeze:
                return prompt_embeds.squeeze(0), pooled_prompt_embeds.squeeze(0)
            else:
                return prompt_embeds, pooled_prompt_embeds
    else:  # SD1.5 or SD2.1
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer",
            use_fast=True,
        )

        text_encoder = CLIPTextModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
        ).cuda()
        text_encoder.requires_grad_(False)

        def enc_prompt(prompt: str, tokenizer, text_encoder):
            tokens = (
                tokenizer(
                    prompt,
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                .input_ids[0]
                .cuda()
            )
            embeds = text_encoder(tokens.unsqueeze(0))[0].detach().cpu()

            # Clean VRAM
            del tokenizer
            del text_encoder
            th.cuda.empty_cache()

            if squeeze:
                return embeds.squeeze(0)
            else:
                return embeds

    return enc_prompt(prompt, tokenizer, text_encoder)


def encode_prompts(model_name: str, prompt_dict: Dict[Any, str], pad_tokens: bool = True) -> Dict[Any, Tuple[str, th.Tensor]]:
    """Encode a dictionary of prompts (key=img_file_name, value=prompt)

    :param model_name: Huggingface model name to encode prompt for.
    :param prompt_dict: A dictionary containing prompts as values.
    :param pad_tokens: If false, remove padding from prompts (recommended for instance prompts but not necessarilly
        for global prompts).
    :return: Dict with the same keys, but Tuple of (prompt, encoded prompt (as torch tensor)) as values.
    """
    if "xl" in model_name:
        tokenizer_one = AutoTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer",
            use_fast=True,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer_2",
            use_fast=True,
        )

        text_encoder_one = CLIPTextModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
        ).cuda()
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            model_name,
            subfolder="text_encoder_2",
        ).cuda()
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)

        def encode_prompt(prompt: str):
            prompt_embeds_list = []

            for tokenizer, text_encoder in zip(
                [tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two]
            ):
                tokens = tokenizer(
                    prompt,
                    padding="max_length" if pad_tokens else "do_not_pad",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.cuda()

                # If not padding tokens remove startoftext and endoftext tokens (do they have any meaning for SD anyway?)
                if not pad_tokens:
                    raise NotImplementedError("Check tokenizer for the correct number of padding tokens")
                    tokens = tokens[tokens != 49406 and tokens != 49407]

                prompt_embeds = text_encoder(tokens, output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = th.concat(prompt_embeds_list, dim=-1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)

            return prompt, prompt_embeds.squeeze(0).cpu(), pooled_prompt_embeds.squeeze(0).cpu()
    else:  # SD1.5 or SD2.1
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer",
            use_fast=True,
        )

        text_encoder = CLIPTextModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
        ).cuda()
        text_encoder.requires_grad_(False)

        def encode_prompt(prompt: str):
            tokens = (
                tokenizer(
                    prompt,
                    max_length=tokenizer.model_max_length,
                    padding="max_length" if pad_tokens else "do_not_pad",
                    truncation=True,
                    return_tensors="pt",
                )
                .input_ids[0]
                .cuda()
            )

            # If not padding tokens, remove startoftext and endoftext tokens (do they have any meaning for SD anyway?)
            if not pad_tokens:
                mask = ~th.isin(tokens, th.tensor([49406, 49407], device=tokens.device))
                tokens = tokens[mask]

            embeds = text_encoder(tokens.unsqueeze(0))[0].detach().squeeze(0).cpu()

            return prompt, embeds

    prompt_enc = {}
    for img_name, prompt in tqdm(prompt_dict.items(), desc="Encoding prompts..."):
        prompt_enc[img_name] = encode_prompt(prompt)

    if "xl" in model_name:
        del tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two
    else:
        del tokenizer, text_encoder
    th.cuda.empty_cache()

    return prompt_enc


def move_tensors_to_device(
        data: Union[th.Tensor, Dict[Any, Any], List[Any], Tuple[Any], Set[Any]],
        device: th.device
) -> Union[th.Tensor, Dict[Any, Any], List[Any], Tuple[Any], Set[Any]]:
    """Recursively move all torch.Tensor objects in a dictionary or other nested structures to the specified device.

    :param data: The input data structure (dict, list, tuple, etc.).
    :param device: The target device (e.g., "cuda", "cpu").

    :return: The data structure with all torch.Tensor objects moved to the specified device.
    """
    if isinstance(data, th.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_tensors_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_device(item, device) for item in data)
    elif isinstance(data, set):
        return {move_tensors_to_device(item, device) for item in data}
    else:
        return data


def change_tensors_to_dtype(
        data: Union[th.Tensor, Dict[Any, Any], List[Any], Tuple[Any], Set[Any]],
        dtype: th.dtype
) -> Union[th.Tensor, Dict[Any, Any], List[Any], Tuple[Any], Set[Any]]:
    """Recursively change dtype of all torch.Tensor objects in a dictionary or other nested structures.

    :param data: The input data structure (dict, list, tuple, etc.).
    :param dtype: The target dtype (e.g., th.float16, th.int7, etc.).

    :return: The data structure with all torch.Tensor objects adapted to the new dtype.
    """
    if isinstance(data, th.Tensor):
        return data.to(dtype)
    elif isinstance(data, dict):
        return {key: change_tensors_to_dtype(value, dtype) for key, value in data.items()}
    elif isinstance(data, list):
        return [change_tensors_to_dtype(item, dtype) for item in data]
    elif isinstance(data, tuple):
        return tuple(change_tensors_to_dtype(item, dtype) for item in data)
    elif isinstance(data, set):
        return {change_tensors_to_dtype(item, dtype) for item in data}
    else:
        return data
