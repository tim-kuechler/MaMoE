from typing import List, Dict, Tuple

import torch as th
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.utils import logging
from torch.nn.utils.rnn import pad_sequence

from mamoe.models.mamoe.utils import BasicTransformerBlockForward

logger = logging.get_logger(__name__)


class RegionMaskedCrossAttnProcessor:
    """A CrossAttnProcessor that enables using masked attentions."""

    @staticmethod
    def prepare_attention(
            size_dim: int,
            dtype: th.dtype,
            attention_mask: List[Dict[int, th.Tensor]],
            encoder_hidden_states: List[Dict[int, th.Tensor]],
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Prepares the provided cross-attention masks and prompt encodings and pads then accross the batch to match the maximum sequence
        length.

        :param size_dim: The hidden size of this block.
        :param dtype: The dtype of this block's layers.
        :param attention_mask: The cross_attention masks.
        :param encoder_hidden_states: The encoded prompts for each attention mask.
        :return: Tuple of prepared cross-attention masks and prompt encodings.
        """
        # Get attention maps and encoder states
        attention_mask = [attn_mask_dict[size_dim] for attn_mask_dict in attention_mask]
        encoder_hidden_states = [enc_states_dict[size_dim] for enc_states_dict in encoder_hidden_states]

        # Pad encoder_hidden_states and attention mask (as every image has a different number of objects)
        encoder_hidden_states = pad_sequence(encoder_hidden_states, batch_first=True).to(dtype)  # (B, sum(d_i), 1024)
        attention_mask = pad_sequence([attn_mask.permute(2, 0, 1) for attn_mask in attention_mask],
                                      batch_first=True, padding_value=-10000.0).permute(0, 2, 3, 1).to(dtype)  # (B, head_dim, h*w, sum(d_i))

        return attention_mask, encoder_hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: th.Tensor,
        encoder_hidden_states: List[Dict[int, th.Tensor]],  # List of enc. text (n_i, 77, 1024). One text per object
        attention_mask: List[Dict[int, th.Tensor]],  # List of binary masks (n_i, h, w), with n_i objects per image
    ) -> th.Tensor:
        """Computes the attention"""
        assert isinstance(encoder_hidden_states, list) and isinstance(attention_mask, list)

        batch_size, size_dim, inner_dim = hidden_states.shape
        head_dim = inner_dim // attn.heads

        # Prepare attention mask and encoder hidden states
        attention_mask, encoder_hidden_states = self.prepare_attention(size_dim, hidden_states.dtype,
                                                                       attention_mask, encoder_hidden_states)

        # Compute query
        query = attn.to_q(hidden_states)
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Transform key and value
        key = attn.to_k(encoder_hidden_states)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        value = attn.to_v(encoder_hidden_states)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Run masked attention
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def setup_region_masked_attn(unet: UNet2DConditionModel):
    """Inject the Regional Masked Attention layers into Huggingface UNet's layers.

    :param unet: UNet.
    """
    unet = replace_call_method_for_unet(unet)

    attn_procs = {}
    for name, layer in unet.attn_processors.items():
        is_self_attention = 'attn1' in name
        if is_self_attention:
            attn_procs[name] = layer
        else:
            attn_procs[name] = RegionMaskedCrossAttnProcessor()

    unet.set_attn_processor(attn_procs)


def replace_call_method_for_unet(model) -> th.nn.Module:
    """Recursively replace all BasicTransformerBlock forwards.

    :param model: Model or model layer.
    """
    for name, layer in model.named_children():
        if layer.__class__.__name__ == 'BasicTransformerBlock':
            from diffusers.models.attention import BasicTransformerBlock
            layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)

        replace_call_method_for_unet(layer)

    return model
