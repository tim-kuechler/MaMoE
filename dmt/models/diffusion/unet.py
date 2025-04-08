import random
from typing import Any, Dict, Union, Optional, List, Tuple

import torch as th
import torch.nn as nn
from diffusers.models import UNet2DConditionModel, UNetControlNetXSModel
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import load_model

from dmt.utils import RankedLogger, encode_prompt

log = RankedLogger(__name__, rank_zero_only=True)


class UNetWrapper(nn.Module):
    """A wrapper around the UNet part of a SD (SD1.5, SD2.1 or SDXL).

    Computes for each model (esp. SDXL) the necessary conditionings (e.g. crop coordinates, ...). Runs the forward
    pass on the model and returns model prediction.
    """

    def __init__(
            self,
            unet: Union[PeftModel, UNet2DConditionModel, UNetControlNetXSModel],
            model_name: str,
            cond_mechanism: str,
            cond_key: str = "pixel_values",
            cn_cond_scale: float = 1.0,
            cn_dropout: float = 0.0,
            txt_dropout: float = 0.0,
            zero_txt_emb: Optional[th.Tensor] = None,
    ) -> None:
        """Init a UNet wrapper for SD.

        :param unet: The unet model to wrap around.
        :param model_name: The huggingface name of the loaded model.
        :param cond_mechanism: The conditioning mechanism the model was initialized with.
        :param cond_key: The key to get the conditioning input from batch.
        :param cn_cond_scale: How much the CNXS affects the base model outputs.
        :param cn_dropout: The dropout probability for the ControlNet conditioning.
        :param txt_dropout: The dropout probability for the txt conditioning.
        """
        super().__init__()

        self.unet = unet
        self.model_name = model_name
        self.cond_mechanism = cond_mechanism
        self.cond_key = cond_key
        self.cn_cond_scale = cn_cond_scale
        self.cn_dropout = cn_dropout
        self.txt_dropout = txt_dropout
        self.zero_txt_emb = zero_txt_emb

    def do_cn_dropout(self, cond: th.Tensor, dropout_overwrite: Optional[float] = None) -> Optional[th.Tensor]:
        """Drops out (replaces with 0s) the control with a certain probability.

        :param cond: The conditioning tensor (e.g. depth map).
        :param dropout_overwrite: Overwrite the dropout probability from self.cn_dropout.
        :return: A tensor of same shape and dtype as input tensor but with 0s only, if the conditioning was dropped out,
            otherwise the original conditioning.
        """
        if self.cond_mechanism == "none":
            return None

        dropout = self.cn_dropout if dropout_overwrite is None else dropout_overwrite
        if dropout > 0.0 and random.uniform(0., 1.) <= dropout:
            return th.zeros_like(cond).to(self.unet.dtype)
        return cond.to(self.unet.dtype)

    def do_txt_dropout(
            self,
            txt_emb,
            txt_emb_2: Optional[th.Tensor] = None,
            dropout_overwrite: Optional[float] = None
    ) -> Union[Tuple[th.Tensor, bool], Tuple[th.Tensor, th.Tensor, bool]]:
        """Drop out the text conditioning to train with Classifier-free guidance according to a probability.
        For SD 1.5 and 2.1 returns the encoded empty string "", for SDXL returns 0s (as with original SD).

        :param txt_emb: The first text encoding
        :param txt_emb_2: The second text encoding (only SDXL)
        :param dropout_overwrite: Overwrite the dropout probability from self.txt_dropout.
        :return: For SD 1.5 and 2.1 encoding of empty prompt, for SDXL two encodings with 0s, if the text is dropped
            out. Otherwise, return original encodings. The bool in the output specifies if the text was dropped out.
        """
        dropout = self.txt_dropout if dropout_overwrite is None else dropout_overwrite
        if dropout > 0.0 and random.uniform(0., 1.) <= dropout:
            if "xl" in self.model_name:
                assert txt_emb_2 is not None
                return th.zeros_like(txt_emb), th.zeros_like(txt_emb_2), True
            else:
                assert self.zero_txt_emb is not None
                return self.zero_txt_emb.clone().to(txt_emb.device).repeat(txt_emb.shape[0], 1, 1), True
        if txt_emb_2 is None:
            return txt_emb, False
        else:
            return txt_emb, txt_emb_2, False

    def forward(
        self,
        noisy_latent: th.Tensor,
        timesteps: th.Tensor,
        batch: Dict[str, Any],
        cn_dropout: Optional[float] = None,
        txt_dropout: Optional[float] = None,
        apply_control: bool = True,
    ) -> th.Tensor:
        """Runs a forward on SD UNet.
        Calculates the required text and time embeddings required for SD.
        :param noisy_latent: A latent image with noise already added.
        :param timesteps: Timesteps used to add the noise.
        :param batch: The batch containing the information to compute the conditionings.
        :param cn_dropout: Overwrites the default dropout probability for the control net conditioning.
        :param txt_dropout: Overwrites the default dropout probability for the text conditioning.
        :param apply_control: If false, do not apply the control.
        :return: Model prediction.
        """
        # Prepare Conditioning
        cond = self.do_cn_dropout(batch[self.cond_key], cn_dropout)
        cond_kwargs = {}
        if self.cond_mechanism == "cn":
            cond = cond / 255.0
            cond_kwargs = {
                "controlnet_cond": cond,
                "conditioning_scale": self.cn_cond_scale,
                "apply_control": apply_control,
            }
        elif self.cond_mechanism == "concat":
            noisy_latent = th.cat((noisy_latent, cond), dim=1)

        # Model forward
        if "xl" not in self.model_name:
            # SD1.5 and SD2.1
            prompt_emb = batch["prompt_emb"] if "prompt_emb" in batch \
                else self.zero_txt_emb.clone().to(noisy_latent.device).repeat(noisy_latent.shape[0], 1, 1)
            prompt_emb, dropped_out = self.do_txt_dropout(prompt_emb, dropout_overwrite=txt_dropout)
            prompt_emb = prompt_emb.to(noisy_latent.dtype)

            # Test if model uses masked cross attention
            region_dict = {}
            if "cross_attn_prompts" in batch:
                assert "cross_attn_mask" in batch
                if dropped_out:
                    prompt_emb_zero = prompt_emb[0, :, :]
                    prompt_emb = None
                    cross_attn_prompts = batch["cross_attn_prompts"]
                    cross_attn_mask = batch["cross_attn_mask"]
                    lat_h, lat_w = noisy_latent.shape[2], noisy_latent.shape[3]
                    for i in range(len(cross_attn_prompts)):
                        for size in ((lat_h, lat_w), (lat_h // 2, lat_w // 2), (lat_h // 4, lat_w // 4), (lat_h // 8, lat_w // 8)):
                            cross_attn_prompts[i][size[0] * size[1]] = prompt_emb_zero
                            cross_attn_mask[i][size[0] * size[1]] = th.zeros_like(cross_attn_mask[i][size[0] * size[1]])[:, :, 0].unsqueeze(dim=-1)

                    region_dict["cross_attn_prompts"] = cross_attn_prompts
                    region_dict["cross_attn_mask"] = cross_attn_mask
                else:
                    prompt_emb = None
                    region_dict["cross_attn_prompts"] = batch["cross_attn_prompts"]
                    region_dict["cross_attn_mask"] = batch["cross_attn_mask"]

            # If model uses MoE pass the expert mask
            if "moe_mask" in batch:
                region_dict["moe_mask"] = batch["moe_mask"]

            # If model uses region self attn pass the mask
            if "self_attn_mask" in batch:
                region_dict["self_attn_mask"] = batch["self_attn_mask"]

            model_pred = self.unet(
                noisy_latent,
                timesteps,
                encoder_hidden_states=prompt_emb,
                cross_attention_kwargs=region_dict if len(region_dict) > 0 else None,
                **cond_kwargs,
            ).sample
        else:
            raise NotImplementedError("The code below was never really tested for SDXL and probably does now work"
                                      "with all features or is incorrect. Use at your own risk and draw inspiration"
                                      "from above for SD2.1 when fixing it.")

            # SDXL
            prompt_emb = batch["prompt_emb"] if "prompt_emb" in batch \
                else self.zero_txt_emb[0].clone().to(noisy_latent.device).repeat(noisy_latent.shape[0], 1, 1)
            prompt_emb_2 = batch["prompt_emb_2"] if "prompt_emb_2" in batch \
                else self.zero_txt_emb[1].clone().to(noisy_latent.device).repeat(noisy_latent.shape[0], 1)
            prompt_emb, prompt_emb_2 = self.do_txt_dropout(prompt_emb, prompt_emb_2, dropout_overwrite=txt_dropout)

            orig_size = batch["orig_size"]
            cropped_size = batch["cropped_size"]
            crop_top_left = batch["crop_coords_top_left"]

            add_time_ids = _get_add_time_ids(orig_size, crop_top_left, cropped_size, dtype=prompt_emb.dtype)
            add_time_ids = add_time_ids.to(prompt_emb.device)
            added_cond_kwargs = {"text_embeds": prompt_emb_2, "time_ids": add_time_ids}

            model_pred = self.unet(
                noisy_latent,
                timesteps,
                encoder_hidden_states=prompt_emb,
                added_cond_kwargs=added_cond_kwargs,
                **cond_kwargs,
            ).sample

        return model_pred


def init_unet(
    model_name: str,
    cond_mechanism: str = "none",
    cn_cond_scale: float = 1.0,
    cond_key: str = "pixel_values",
    cn_block_out_channels: List[int] = (4, 8, 16, 16),
    cn_attn_head_dim: List[int] = (1, 1, 2, 2),
    cn_learn_time: bool = False,
    cn_conditioning_channels: int = 3,
    use_lora: bool = False,
    lora_rank: int = 128,
    lora_alpha: int = 128,
    use_dora: bool = False,
    only_decoder: bool = False,
    cn_dropout: float = 0.0,
    txt_dropout: float = 0.0,
    keep_base_frozen: bool = False,
    ckpt_path: str = None,
) -> UNetWrapper:
    """Initializes the UNet part of SD and applies LoRA.

    The loaded model will be returned in train mode and with LoRA adapter.

    :param model_name: Model name of the pretrained base model.
    :param cond_mechanism: "none", "cn" for control net xs or "concat" for concatening conditioning.
    :param cn_cond_scale: How much the control model affects the base model outputs.
    :param cond_key: The key in the batch dictionary to get the CNXS conditioning.
    :param cn_block_out_channels: The ControlNetXS block_out_channels
    :param cn_attn_head_dim: The ControlNetXS attention head dimension.
    :param cn_learn_time: If true, ControlNetXS learns its own timestep encoding.
    :param cn_conditioning_channels: The number of channels in the conditioning. 1 might not work, so rather consider
        duplicating the conditioning channels 3 times in this case (e.g. for depth).
    :param use_lora: If True initializes UNet with LoRA on SD Base (not on CNXS).
    :param lora_rank: LoRA rank.
    :param lora_alpha: LoRA alpha.
    :param use_dora: If `True` uses DoRA instead of LoRA (https://arxiv.org/abs/2402.09353).
    :param only_decoder: If True make only the decoder (+conv_out) of SD Base or LoRA trainable.
    :param cn_dropout: The ratio at which the CNXS conditioning is dropped during training.
    :param txt_dropout: The ratio at which the txt conditioning is dropped during training.
    :param keep_base_frozen: If true, sets the parameters of the base model as non-trainable (e.g. in the case of LoRA
        or CNXS).
    :param ckpt_path: An optional ckpt_path to load when initing the UNet.
    :return: UNetWrapper to train.
    """
    assert cond_mechanism in ("none", "cn", "concat"), f"Conditioning mechanism {cond_mechanism} not available."

    # Setup conditioning mechanism and init UNet
    if cond_mechanism != "concat":
        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
    else:
        # Load UNet with concat conditioning
        # Init UNet with 8 in_channels (conv_in will be initialized with random weight)
        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet",
                                                    in_channels=8, low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True)

        # Get original conv_in weights
        conv_in_weight = (UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
                          .conv_in.weight.data.to(unet.device))  # (C, 4, 3, 3)

        # Double the weight channels and half weight values.
        conv_in_weight = th.cat((conv_in_weight, conv_in_weight), dim=1) / 2  # (C, 8, 3, 3)

        # Replace conv_in weights in unet
        unet.conv_in.weight = nn.Parameter(conv_in_weight)

    # Load optional checkpoint
    if ckpt_path is not None:
        log.info(f"Loading checkpoint {ckpt_path}..")
        load_model(unet, ckpt_path, True)

    # Add ControlNetXS
    if cond_mechanism == "cn":
        unet.config.sample_size = 64
        unet.sample_size = 64
        unet = UNetControlNetXSModel.from_unet(
            unet,
            ctrl_block_out_channels=list(cn_block_out_channels),
            ctrl_optional_kwargs={
                "learn_time_embedding": cn_learn_time,
                "conditioning_channels": cn_conditioning_channels,
                "num_attention_heads": list(cn_attn_head_dim),
            }
        )

    # Add LoRA
    if use_lora:
        # Extract layers to apply LoRA to
        target_modules = []
        grep = [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "conv",
            "conv1",
            "conv2",
            "conv_in",
            "conv_shortcut",
            "conv_out",
            "proj_out",
            "proj_in",
            "ff.net.2",
            "ff.net.0.proj",
        ]

        for n, p in unet.named_parameters():
            if ("bias" in n or "norm" in n) or (cond_mechanism == "cn" and ("base" not in n and "up_blocks" not in n)):
                continue
            for pattern in grep:
                # Always add LoRA to decoder and conv_out
                if pattern in n and ("up_blocks" in n or "conv_out" in n):
                    target_modules.append(n.replace(".weight", ""))
                    break
                # Add LoRA also encoder + anything else if only_decoder=False
                elif pattern in n and not only_decoder \
                        and "control_to_base_for_conv_in" not in n\
                        and not (cond_mechanism == "concat" and "conv_in" in n):
                    target_modules.append(n.replace(".weight", ""))
                    break

        # Apply LoRA to UNet
        lora_conf = LoraConfig(
            r=lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            use_dora=use_dora,
        )
        unet = get_peft_model(unet, lora_conf, adapter_name="base_default")

    # Handle setting require_grad=True/False for the correct layers
    unet.requires_grad_(False)

    if cond_mechanism == "cn":
        # Set CN parameters to true (sets ALL other parameters (LoRA and SD Base) to false)
        if use_lora:
            unet.base_model.model.freeze_unet_params()
        else:
            unet.freeze_unet_params()

    if not use_lora:
        if not keep_base_frozen:
            # Sets Base SD parameters to true
            if not only_decoder:
                # Train All Layers
                unet.requires_grad_(True)
            else:
                # Train only decoder + conv_out
                for n, p in unet.named_parameters():
                    if "up_blocks" in n or "conv_out" in n:
                        p.requires_grad = True
    else:
        # Sets LoRA parameters to true
        # No distinction betw. full and decoder-only training necessary because if decoder-only then LoRA
        # layers for encoder aren't even created in the first place.
        for n, p in unet.named_parameters():
            if "lora" in n and "base_default" in n:
                p.requires_grad = True

    if cond_mechanism == "concat":
        if use_lora:
            unet.base_model.model.conv_in.weight.requires_grad = True
        else:
            unet.conv_in.weight.requires_grad = True

    unet.train()

    # Calculating zero embedding for CFG dropout
    zero_txt_emb = encode_prompt(model_name, "", False)

    return UNetWrapper(unet, model_name, cond_mechanism, cond_key, cn_cond_scale, cn_dropout, txt_dropout, zero_txt_emb)


def _get_add_time_ids(
    orig_size: List[th.Tensor], crop_top_left: List[th.Tensor], target_size: List[th.Tensor], dtype: th.dtype
) -> th.Tensor:
    """Transforms the additional time ids for SDXL into a tensor (i.e. orig. size, crop coords and
    target size).
    """
    orig_size = [(int(a.item()), int(b.item())) for a, b in zip(orig_size[0], orig_size[1])]
    crop_top_left = [(int(a.item()), int(b.item())) for a, b in zip(crop_top_left[0], crop_top_left[1])]
    target_size = [(int(a.item()), int(b.item())) for a, b in zip(target_size[0], target_size[1])]

    add_time_ids = [(a, b, c, d, e, f) for (a, b), (c, d), (e, f)
                                    in zip(orig_size, crop_top_left, target_size)]
    add_time_ids = th.tensor(add_time_ids, dtype=dtype)

    return add_time_ids
