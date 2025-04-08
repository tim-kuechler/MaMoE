from copy import deepcopy
from typing import Dict, List, Union, Tuple

import torch as th
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.attention import FeedForward
from diffusers.utils import logging
from torch import nn

from dmt.models.mamoe.utils import BasicTransformerBlockForward

logger = logging.get_logger(__name__)


class MoEFeedForward(nn.Module):
    """Class-aware Mixture of Experts layer that is applied to the FeedForward layers of every SD Transformer block."""

    def __init__(self, feed_forward: FeedForward, num_experts: int):
        """Init a MoEFeedForward.

        :param feed_forward: The original FeedForward layer of the block.
        :param num_experts: The number of experts in this layer.
        """
        super().__init__()
        self.num_experts = num_experts
        ffs = [deepcopy(feed_forward) for _ in range(num_experts)]

        # project in
        self.linear_in_weight = nn.Parameter(th.stack([ff.net[0].proj.weight for ff in ffs]).clone().contiguous())  # (n, 2560, 320)
        self.linear_in_bias = nn.Parameter(th.stack([ff.net[0].proj.bias for ff in ffs]).clone().contiguous())  # (n, 2560)

        # project out
        self.linear_out_weight = nn.Parameter(th.stack([ff.net[2].weight for ff in ffs]).clone().contiguous())  # (n, 320, 1280)
        self.linear_out_bias = nn.Parameter(th.stack([ff.net[2].bias for ff in ffs]).clone().contiguous())  # (n, 320)


    def forward(self, hidden_states: th.Tensor) -> th.Tensor:
        # Linear in
        hidden_states = th.einsum('nbcd,nfd->nbcf', hidden_states, self.linear_in_weight)  # (n, b, c, 320/640/...)
        hidden_states += self.linear_in_bias.view(self.num_experts, 1, 1, hidden_states.shape[3])  # (n, b, c, 2560/5120/...)

        # Gelu
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = hidden_states * F.gelu(gate)

        # Linear out
        hidden_states = th.einsum('nbcd,nfd->nbcf', hidden_states, self.linear_out_weight)  # (n, b, c, 2560/5120/...)
        hidden_states += self.linear_out_bias.view(self.num_experts, 1, 1, hidden_states.shape[3])  # (n, b, c, 320/640/...)

        return hidden_states


class MoE(nn.Module):
    """The wrapper around the MoE layer. This class is responsible for handling the dispatch and combine masks."""

    def __init__(self, feed_forward: FeedForward, num_experts):
        """Init a MoE.

        :param feed_forward: The original FeedForward layer of the block.
        :param num_experts: The number of experts in this layer.
        """
        super().__init__()

        self.num_experts = num_experts
        self.experts = MoEFeedForward(feed_forward, num_experts)


    @staticmethod
    def prepare_masks(
            size_dim: int,
            dtype: th.dtype,
            mask: List[Tuple[Dict[int, th.Tensor], Dict[int, th.Tensor]]],
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Prepare the provided dispatch and combine masks.

        :param size_dim: The hidden size of this block.
        :param dtype: The dtype of this block's layers.
        :param mask: The dispatch and combine masks.
        """
        assert mask is not None, "No MoE mask provided"

        # Get attention maps and encoder states
        dispatch_mask = th.stack([mask_dict[0][size_dim] for mask_dict in mask]).to(dtype)
        combine_mask = th.stack([mask_dict[1][size_dim] for mask_dict in mask]).to(dtype)

        return dispatch_mask, combine_mask

    def forward(
            self,
            hidden_states: th.Tensor,
            mask: List[Tuple[Dict[int, th.Tensor], Dict[int, th.Tensor]]],
    ) -> th.Tensor:
        dispatch_mask, combine_mask = self.prepare_masks(hidden_states.shape[1], hidden_states.dtype, mask)

        # Split input for experts by provided mask
        hidden_states = th.einsum('bcd,bcn->nbcd', hidden_states, dispatch_mask)  # (n, b, c, d)

        # Run experts
        hidden_states = self.experts(hidden_states)

        # Merge expert outputs
        hidden_states = th.einsum('nbcd,bcn->bcd', hidden_states, combine_mask)  # (b, c, d)

        return hidden_states


def _check_layer(layers: List[int], name: str) -> bool:
    """ Extracts the blocks number from its name and checks if its in the provided list.

    :param layers: A list of integers describing the block numbers MoE should be applied to.
    :param name: The name of the block.
    :return: True, if MoE should be applied to the block, False otherwise.
    """
    if len(layers) == 0:
        return True

    if "mid_block" in name:
        # When layers are set, always assume that mid-block should be skipped
        return False
    elif "blocks" in name:
        block_number = int(name.split(".")[2])

        if "down_blocks" in name:
            return block_number in layers
        elif "up_blocks" in name:
            return 3 - block_number in layers

    return False


def setup_moe(
        unet: Union[UNet2DConditionModel, nn.Module],
        num_experts: int,
        layers: List[int] = [],  # Empty list means all layers
        prev_name: str = ""
) -> None:
    """Inject the MoE layers into the Huggingface UNet's layers.

    :param unet: UNet.
    :param num_experts: The number of experts in each layer.
    :param layers: A list of integers to which the MoE layer should be applied.
    :param prev_name: The previous layer name (this method is called recursively).
    """
    for name, layer in unet.named_children():
        if layer.__class__.__name__ == 'BasicTransformerBlock' and "ctrl" not in prev_name and _check_layer(layers, prev_name):
            from diffusers.models.attention import BasicTransformerBlock
            layer.ff = MoE(layer.ff, num_experts)
            layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)

        setup_moe(layer, num_experts, layers, f"{prev_name}.{name}")
