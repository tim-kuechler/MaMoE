import types
from typing import Any, Dict, Optional, Tuple, Union

import torch as th
import torch.nn as nn
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

from dmt.models.diffusion import UNetWrapper, init_unet


class PerceptualFeatureExtractor(nn.Module):
    """Extract the hidden states of the UNet forward after the mid-block. These features can be used
    as a perceptual loss.
    """

    def __init__(self, model_name: str, ckpt_path: Optional[str] = None, *args, **kwargs):
        """Init a PerceptualFeatureExtractor.

        :param model_name: Huggingface model repository name. Should likely be the same as for the main model
            that is trained.
        :param ckpt_path: An optional checkpoint to initialize the model from (e.g. if a fine-tuned model should be
            used as an extractor).
        """
        super().__init__(*args, **kwargs)
        # Init UNet as backend for the extractor
        self.unet_wrapper: UNetWrapper = init_unet(model_name, ckpt_path=ckpt_path,
                                                   cond_mechanism="none", use_lora=False)
        self.unet_wrapper.unet.forward = types.MethodType(UNet2DConditionModelForward, self.unet_wrapper.unet)

        # Delete every layer but conv_in, down_blocks and mid_block
        del self.unet_wrapper.unet.up_blocks
        del self.unet_wrapper.unet.conv_norm_out
        del self.unet_wrapper.unet.conv_act
        del self.unet_wrapper.unet.conv_out

    def forward(self, x: th.Tensor, timesteps: th.Tensor, batch: Dict[str, Any]):
        return self.unet_wrapper(x, timesteps, batch)


def UNet2DConditionModelForward(
    self,
    sample: th.Tensor,
    timestep: Union[th.Tensor, float, int],
    encoder_hidden_states: th.Tensor,
    class_labels: Optional[th.Tensor] = None,
    timestep_cond: Optional[th.Tensor] = None,
    attention_mask: Optional[th.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    added_cond_kwargs: Optional[Dict[str, th.Tensor]] = None,
    encoder_attention_mask: Optional[th.Tensor] = None,
    return_dict: bool = True,
) -> Union[UNet2DConditionOutput, Tuple]:
    """Adapted UNet2DConditionModel forward method that only runs till the mid-block and returns the output after
    the mid-block"""
    default_overall_up_factor = 2**self.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            # Forward upsample size to force interpolation output size.
            forward_upsample_size = True
            break

    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb = self.time_embedding(t_emb, timestep_cond)
    aug_emb = None

    class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
    if class_emb is not None:
        if self.config.class_embeddings_concat:
            emb = th.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    aug_emb = self.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )
    if self.config.addition_embed_type == "image_hint":
        aug_emb, hint = aug_emb
        sample = th.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    encoder_hidden_states = self.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    # 2. pre-process
    sample = self.conv_in(sample)

    # 2.5 GLIGEN position net
    if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
        cross_attention_kwargs = cross_attention_kwargs.copy()
        gligen_args = cross_attention_kwargs.pop("gligen")
        cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

    # 3. down
    down_block_res_samples = (sample,)
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}

            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        down_block_res_samples += res_samples

    # 4. mid
    if self.mid_block is not None:
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = self.mid_block(sample, emb)

    if not return_dict:
        return (sample,)

    return UNet2DConditionOutput(sample=sample)
