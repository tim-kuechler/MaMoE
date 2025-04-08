import os
from collections import defaultdict
from itertools import compress
from typing import Any, Dict, List, Tuple, Optional

import torch as th
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from diffusers.utils import logging
from einops import rearrange
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2.functional import horizontal_flip, crop

from mamoe.data import BaseDataset
from mamoe.utils import encode_prompts

logger = logging.get_logger(__name__)


def _resize_attn_mask(
        attn_mask: th.Tensor,
        height: int,
        width: int,
        resize_mode: str = "nearest",
        pool_threshold: float = 0.5,
) -> Tuple[th.Tensor, Optional[th.Tensor]]:
    """Resizes a tensor with binary masks of shape (N, h, w) to a new size.

    :param attn_mask: The binary mask.
    :param height: The target height.
    :param width: The target width.
    :param resize_mode: The method to use for resizing. Either "nearest" or "avg_pool".
    :param pool_threshold: The threshold value for the average pool resizing.
    """
    assert resize_mode in ("nearest", "avg_pool")

    if resize_mode == "nearest":
        return TF.resize(attn_mask, [height, width], TF.InterpolationMode.NEAREST_EXACT).bool(), None
    else:
        # Compute pooling kernel size based on the target shape
        _, h, w = attn_mask.shape
        kernel_size = (h // height, w // width)

        # Apply average pooling
        pooled = F.avg_pool2d(attn_mask, kernel_size=kernel_size)

        # Apply threshold to get the binary mask
        return (pooled > pool_threshold).bool(), pooled


def _prepare_cross_attention(
        height: int,
        width: int,
        num_heads: int,
        attention_mask: List[th.Tensor],
        text_encoding: List[th.Tensor],
        resize_mode: str = "nearest",
        pool_threshold: float = 0.5,
) -> Tuple[th.Tensor, th.Tensor]:
    """Prepares the cross attention masks for Masked Attention.

    :param height: The target height.
    :param width: The target width.
    :param num_heads: The number of attention heads in the target block.
    :param attention_mask: A List of attention masks.
    :param text_encoding: A List of the corresponding text encodings.
    :param resize_mode: The method to use for resizing. Either "nearest" or "avg_pool".
    :param pool_threshold: The threshold value for the average pool resizing.
    :return: Prepared attention masks.
    """
    # TODO Remove masks in regions where the ego vehicle is (don't forget to check flip)
    assert len(attention_mask) == len(text_encoding)

    # Clone tensors
    attention_mask = th.stack(attention_mask)
    attention_mask = attention_mask.clone().float()
    text_encoding = [enc.clone() for enc in text_encoding]

    # Resize attention masks
    attention_mask, _ = _resize_attn_mask(attention_mask, height, width, resize_mode, pool_threshold)

    # Filter attention masks with 0s only (+ corresponding text encodings)
    non_zero_masks = attention_mask.any(dim=(1, 2))
    if non_zero_masks.max() != 0:
        attention_mask = attention_mask[non_zero_masks]
        text_encoding = list(compress(text_encoding, non_zero_masks))

    # Combine masks that have the same text encoding (e.g. all masks representing a "car")
    attn_mask_dict = defaultdict(lambda: th.zeros((height, width)))
    unique_text_encodings = []
    for text_enc, attn_mask in zip(text_encoding, attention_mask):
        key = tuple(text_enc.flatten().tolist()) # Tensor directly cannot be used as a dict key (not hashable)

        if key not in attn_mask_dict:
            unique_text_encodings.append(text_enc)

        # Combine masks by adding them
        attn_mask_dict[key] += attn_mask

    attention_mask = th.stack([attn_mask_dict[tuple(text_enc.flatten().tolist())].bool() for text_enc in unique_text_encodings])
    text_encoding = unique_text_encodings

    # Transform attention masks
    repeats = th.tensor([enc.shape[0] for enc in text_encoding])
    attention_mask = attention_mask.repeat_interleave(repeats, dim=0).permute(1, 2, 0) # (h, w, sum(d_i))
    attention_mask = rearrange(attention_mask, "h w t -> (h w) t")  # (h*w, sum(d_i))
    attention_mask = attention_mask.unsqueeze(dim=0).repeat(num_heads, 1, 1)  # (head_dim, h*w, sum(d_i))
    attention_mask = (1 - attention_mask.float()) * -10000.0
    attention_mask = attention_mask.contiguous()

    # Transform text encodings
    text_encoding = th.cat(text_encoding, dim=0).contiguous()

    return attention_mask, text_encoding


def load_panoptic_image(
        dataset: BaseDataset,
        id: int,
        item: Dict[str, Any],
        panoptic_dir: str,
) -> Tuple[str, th.Tensor, th.Tensor, th.Tensor, int, int]:
    """Loads a panoptic map that was encoded into RGB channels. RG are Instance ID, B is class id.

    :param dataset: The dataset objects with stored file paths.
    :param id: The image id.
    :param item: The item dict as created in the __get_item__ method in the dataset.
    :param panoptic_dir: The folder where the panoptic maps are.
    :return: Tuple with img_name, panoptic_image, instance_ids, class_ids, crop_h, crop_w.
    """
    image_path = dataset.image_paths[id]
    img_name = os.path.basename(image_path)
    panoptic_path = os.path.join(panoptic_dir, dataset.split)
    panoptic_img = read_image(panoptic_path, ImageReadMode.RGB)

    # Crop
    crop_h, crop_w = item["cropped_size"]
    orig_h, orig_w = item["orig_size"]
    top, left = item["crop_coords_top_left"]
    # If aspect ratios of rgb image and panoptic image do not match assume rgb image was random square cropped with (min(h, w), min(h, w)
    if crop_h // crop_w != panoptic_img.shape[1] // panoptic_img.shape[2]:
        assert crop_h == crop_w
        top = top * panoptic_img.shape[1] // orig_h
        left = left * panoptic_img.shape[2] // orig_w
        crop_h = crop_w = min(panoptic_img.shape[1], panoptic_img.shape[2])
        panoptic_img = crop(panoptic_img, top, left, crop_h, crop_w)

    # Flip
    if item["flipped"]:
        panoptic_img = horizontal_flip(panoptic_img)

    # Decode panoptic img to instances and classes
    R, G, class_ids = panoptic_img[0], panoptic_img[1], panoptic_img[2]
    instance_ids = (R.to(th.int32) << 8) + G.to(th.int32)

    return img_name, panoptic_img, instance_ids, class_ids, crop_h, crop_w


def add_dataset_to_prompt(
        dataset: BaseDataset,
        encoder_attention_mask: List[th.Tensor],
        encoder_hidden_states: List[th.Tensor],
) -> Tuple[List[th.Tensor], List[th.Tensor]]:
    """Adds the dataset's name to the prompts.

    :param dataset: The pytorch dataset.
    :param encoder_attention_mask: List of attention masks.
    :param encoder_hidden_states: List of encoded prompts.
    :return: Tuple of encoder_attention_mask, encoder_hidden_states, extended by the dataset name.
    """
    assert hasattr(dataset, "dataset_prompt")
    encoder_hidden_states.insert(0, dataset.dataset_prompt[dataset.dataset_name.lower()][1])
    encoder_attention_mask.insert(0, th.ones_like(encoder_attention_mask[0]))

    return encoder_attention_mask, encoder_hidden_states


def prepare_cross_attn_mask_and_text_encodings(
        encoder_attention_mask: List[th.Tensor],
        encoder_hidden_states: List[th.Tensor],
        h: int,
        w: int,
        resize_mode: str,
        pool_threshold: float = 0.5
) -> Tuple[Dict[int, th.Tensor], Dict[int, th.Tensor]]:
    """Prepares the cross attention masks for Masked Attention for various resolutions of the down- and up-blocks.

    :param encoder_attention_mask: A List of attention masks.
    :param encoder_hidden_states: A List of the corresponding text encodings.
    :param h: The target height.
    :param w: The target width.
    :param resize_mode: The method to use for resizing. Either "nearest" or "avg_pool".
    :param pool_threshold: The threshold value for the average pool resizing.
    :return: Prepared attention masks.
    """
    attention_mask_dict = {}
    text_encoding_dict = {}
    lat_h, lat_w = h // 8, w // 8
    for size in ((lat_h, lat_w, 5), (lat_h // 2, lat_w // 2, 10), (lat_h // 4, lat_w // 4, 20), (lat_h // 8, lat_w // 8, 20)):
        attn_mask, text_enc = _prepare_cross_attention(
            size[0],
            size[1],
            size[2],
            encoder_attention_mask,
            encoder_hidden_states,
            resize_mode=resize_mode,
            pool_threshold=pool_threshold
        )
        attention_mask_dict[size[0] * size[1]] = attn_mask
        text_encoding_dict[size[0] * size[1]] = text_enc

    return attention_mask_dict, text_encoding_dict


def prepare_moe_mask(
        moe_masks: List[th.Tensor],
        h: int,
        w: int,
        resize_mode: str,
        pool_threshold: float = 0.5,
) -> Tuple[Dict[int, th.Tensor], Dict[int, th.Tensor]]:
    """Prepares the dispatch and combine masks for Mixture of Experts.

    :param moe_masks: A List of attention masks.
    :param h: The target height.
    :param w: The target width.
    :param resize_mode: The method to use for resizing. Either "nearest" or "avg_pool".
    :param pool_threshold: The threshold value for the average pool resizing.
    :return: Prepared MoE masks.
    """
    dispatch_mask_dict = {}
    combine_mask_dict = {}
    lat_h, lat_w = h // 8, w // 8
    for size in ((lat_h, lat_w), (lat_h // 2, lat_w // 2), (lat_h // 4, lat_w // 4), (lat_h // 8, lat_w // 8)):
        dispatch_mask = th.stack(moe_masks)
        dispatch_mask = dispatch_mask.clone().float()
        # TODO Pool threshold
        dispatch_mask, combine_mask = _resize_attn_mask(dispatch_mask, size[0], size[1], resize_mode, 0.0)
        dispatch_mask = dispatch_mask.float().reshape(dispatch_mask.shape[0], -1)
        dispatch_mask = dispatch_mask.permute(1, 0)
        dispatch_mask_dict[size[0] * size[1]] = dispatch_mask

        if combine_mask is None:
            counts = dispatch_mask.sum(dim=1, keepdim=True)
            counts[counts == 0] = 1 # avoid zero division
            combine_mask = dispatch_mask / counts
        else:
            combine_mask = combine_mask.float().reshape(combine_mask.shape[0], -1)
            combine_mask = combine_mask.permute(1, 0)
        combine_mask_dict[size[0] * size[1]] = combine_mask

    return dispatch_mask_dict, combine_mask_dict


def encode_dataset_name(
        dataset: BaseDataset,
        encoder_padding: bool = False,
) -> None:
    """Encode the dataset name and add it to the dataset.

    :param dataset: The dataset.
    :param encoder_padding: If false, remove any padding from encoded prompts (recommended).
    """
    dataset_name = dataset.dataset_name.lower()
    dataset_prompt = {dataset_name: dataset_name}
    dataset.dataset_prompt = encode_prompts(dataset.model_name, dataset_prompt, pad_tokens=encoder_padding)


def encode_class_prompts(
        dataset: BaseDataset,
        class_prompts: Dict[int, str],
        encoder_padding: bool = False,
) -> None:
    """Encode class names and add it to the dataset.

    :param dataset: The dataset.
    :param class_prompts: A dictionary with the class id as key and the class name as value.
    :param encoder_padding: If false, remove any padding from encoded prompts (recommended).
    """
    dataset.class_prompts = encode_prompts(dataset.model_name, class_prompts, pad_tokens=encoder_padding)


# Adapted from diffusers.models.attention.BasicTransformerBlock.forward. Method was only changed where marked with ###..
def BasicTransformerBlockForward(
    self,
    hidden_states: th.Tensor,
    attention_mask: Optional[th.Tensor] = None,
    encoder_hidden_states: Optional[th.Tensor] = None,
    encoder_attention_mask: Optional[th.Tensor] = None,
    timestep: Optional[th.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[th.LongTensor] = None,
    added_cond_kwargs: Optional[Dict[str, th.Tensor]] = None,
) -> th.Tensor:
    ### Args for region cross- and self-attention as well as MoE are passed through cross_attention_kwargs
    assert cross_attention_kwargs is not None
    ### End

    if cross_attention_kwargs is not None:
        if cross_attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")

    # Notice that normalization is always applied before the real computation in the following blocks.
    # 0. Self-Attention
    batch_size = hidden_states.shape[0]

    if self.norm_type == "ada_norm":
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.norm_type == "ada_norm_zero":
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
        norm_hidden_states = self.norm1(hidden_states)
    elif self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif self.norm_type == "ada_norm_single":
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
    else:
        raise ValueError("Incorrect norm used")

    if self.pos_embed is not None:
        norm_hidden_states = self.pos_embed(norm_hidden_states)

    # 1. Prepare GLIGEN inputs
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
    gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

    ### Check added_cond_kwargs for a self_attn_mask and pass to self attn
    self_attn_mask = cross_attention_kwargs["self_attn_mask"] if "self_attn_mask" in cross_attention_kwargs else None
    if self_attn_mask is not None:
        self_attn_mask = th.stack([attn_mask[norm_hidden_states.shape[1]] for attn_mask in self_attn_mask]).to(norm_hidden_states.dtype)
    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=None,
        attention_mask=self_attn_mask,
    )
    ### End

    if self.norm_type == "ada_norm_zero":
        attn_output = gate_msa.unsqueeze(1) * attn_output
    elif self.norm_type == "ada_norm_single":
        attn_output = gate_msa * attn_output

    hidden_states = attn_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    # 1.2 GLIGEN Control
    if gligen_kwargs is not None:
        hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

    # 3. Cross-Attention
    if self.attn2 is not None:
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm2(hidden_states, timestep)
        elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm2(hidden_states)
        elif self.norm_type == "ada_norm_single":
            # For PixArt norm2 isn't applied here:
            # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
            norm_hidden_states = hidden_states
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
        else:
            raise ValueError("Incorrect norm")

        if self.pos_embed is not None and self.norm_type != "ada_norm_single":
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        ### Check for cross attn prompts and mask and pass to cross attention
        cross_attn_prompts = cross_attention_kwargs["cross_attn_prompts"] \
            if "cross_attn_prompts" in cross_attention_kwargs else encoder_hidden_states
        cross_attn_mask = cross_attention_kwargs["cross_attn_mask"] \
            if "cross_attn_mask" in cross_attention_kwargs else None
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=cross_attn_prompts,
            attention_mask=cross_attn_mask,
        )
        ### End
        hidden_states = attn_output + hidden_states

    # 4. Feed-forward
    # i2vgen doesn't have this norm 🤷‍♂️
    if self.norm_type == "ada_norm_continuous":
        norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
    elif not self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm3(hidden_states)

    if self.norm_type == "ada_norm_zero":
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if self.norm_type == "ada_norm_single":
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

    if self._chunk_size is not None:
        ### Chunked ff not supported with MoE
        raise NotImplementedError("Chunked ff not supported with MoE")
    else:
        ### Check for Moe mask and pass to FeedForward
        if "moe_mask" in cross_attention_kwargs:
            ff_output = self.ff(norm_hidden_states, cross_attention_kwargs["moe_mask"])
        else:
            ff_output = self.ff(norm_hidden_states)
        ### End

    if self.norm_type == "ada_norm_zero":
        ff_output = gate_mlp.unsqueeze(1) * ff_output
    elif self.norm_type == "ada_norm_single":
        ff_output = gate_mlp * ff_output

    hidden_states = ff_output + hidden_states
    if hidden_states.ndim == 4:
        hidden_states = hidden_states.squeeze(1)

    return hidden_states
