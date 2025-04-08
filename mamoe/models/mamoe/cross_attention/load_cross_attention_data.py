from typing import Any, Dict

import torch as th
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset

from mamoe.data.utils import CollatableItem
from mamoe.data import BaseDataset
from mamoe.models.mamoe.utils import (encode_dataset_name, encode_class_prompts, load_panoptic_image,
                                      prepare_cross_attn_mask_and_text_encodings, add_dataset_to_prompt)

# Yes, I know that handling this stuff via global variable is ugly and probably all the code beneath should be
# done via classes and inheritance
encoder_padding = False
add_dataset_name = True
instance_prompt_path = None
resize_mode = "nearest"
global_prompts = False
panoptic_map_dir = ""
pool_threshold = 0.5


def panoptic_cs_things_prompts_get(dataset: BaseDataset, id: int, item: Dict[str, Any]) -> Dict[str, Any]:
    img_name, panoptic_img, instance_ids, class_ids, crop_h, crop_w = (
        load_panoptic_image(dataset, id, item, panoptic_map_dir))

    # Get text encodings for classes/instances in panoptic map
    encoder_hidden_states = []
    unique_ids = th.unique(instance_ids)
    encoder_attention_mask = [(instance_ids == id) for id in unique_ids]
    instance_classes = [(class_ids * mask).max() for mask in encoder_attention_mask]
    for unique_id, inst_class in zip(unique_ids, instance_classes):
        if inst_class == 19 or inst_class == 0 or unique_id == 0:
            encoder_hidden_state = dataset.prompts[img_name][1]
        elif instance_prompt_path is None or img_name not in dataset.instance_prompts:
            encoder_hidden_state = dataset.class_prompts[int(inst_class.item())][1]
        else:
            encoder_hidden_state = dataset.instance_prompts[img_name][unique_id.item()][1]

        encoder_hidden_states.append(encoder_hidden_state)

    # Add dataset name
    if add_dataset_name:
        encoder_attention_mask, encoder_hidden_states = add_dataset_to_prompt(
            dataset,
            encoder_attention_mask,
            encoder_hidden_states
        )

    # Prepare masks and text encodings
    attention_mask_dict, text_encoding_dict = prepare_cross_attn_mask_and_text_encodings(
        encoder_attention_mask,
        encoder_hidden_states,
        crop_h,
        crop_w,
        resize_mode,
        pool_threshold,
    )

    item["cross_attn_prompts"] = CollatableItem(item=text_encoding_dict)
    item["cross_attn_mask"] = CollatableItem(item=attention_mask_dict)
    item["panoptic_img"] = panoptic_img

    return item


def panoptic_cs_things_prompts_init(dataset: BaseDataset) -> None:
    # Encode class prompts
    class_prompts = {13: "car", 15: "bus", 18: "bicycle", 11: "person", 14: "truck", 16: "train", 17: "motorcycle",
                     12: "rider, cyclist"}
    encode_class_prompts(dataset, class_prompts, encoder_padding)

    # Encode dataset name
    if add_dataset_name:
        encode_dataset_name(dataset, encoder_padding)


def panoptic_cs_all_prompts_get(dataset: BaseDataset, id: int, item: Dict[str, Any]) -> Dict[str, Any]:
    img_name, panoptic_img, instance_ids, class_ids, crop_h, crop_w = (
        load_panoptic_image(dataset, id, item, panoptic_map_dir))

    # Get text encodings for classes/instances in panoptic map
    encoder_hidden_states = []
    unique_ids = th.unique(instance_ids)
    encoder_attention_mask = [(instance_ids == id) for id in unique_ids if id != 0]
    instance_classes = [(class_ids * mask).max() for mask in encoder_attention_mask]
    for unique_id, inst_class in zip(unique_ids[unique_ids != 0], instance_classes):
        if (instance_prompt_path is None or img_name not in dataset.instance_prompts
                or unique_id not in list(dataset.instance_prompts[img_name].keys())):
            encoder_hidden_state = dataset.class_prompts[int(inst_class.item())][1]
        else:
            encoder_hidden_state = dataset.instance_prompts[img_name][unique_id.item()][1]
        encoder_hidden_states.append(encoder_hidden_state)

    # Add dataset name
    if add_dataset_name:
        encoder_attention_mask, encoder_hidden_states = add_dataset_to_prompt(
            dataset,
            encoder_attention_mask,
            encoder_hidden_states
        )

    # Add additional prompts
    if global_prompts:
        assert len(dataset.prompts) > 0, "No prompts encoded, provide a prompts file to the dataset."
        encoder_hidden_states.insert(0, dataset.prompts[img_name][1])
        encoder_attention_mask.insert(0, th.ones_like(encoder_attention_mask[0]))

    # Prepare masks and text encodings
    attention_mask_dict, text_encoding_dict = prepare_cross_attn_mask_and_text_encodings(
        encoder_attention_mask,
        encoder_hidden_states,
        crop_h,
        crop_w,
        resize_mode,
        pool_threshold,
    )

    item["cross_attn_prompts"] = CollatableItem(item=text_encoding_dict)
    item["cross_attn_mask"] = CollatableItem(item=attention_mask_dict)
    item["panoptic_img"] = panoptic_img

    return item


def panoptic_cs_all_prompts_init(dataset: BaseDataset) -> None:
    # Encode class prompts
    class_prompts = {13: "car", 15: "bus", 18: "bicycle", 11: "person", 14: "truck", 16: "train", 17: "motorcylce",
                     0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence", 5: "pole", 6: "traffic light",
                     7: "traffic sign", 8: "vegetation, trees, bushes", 9: "terrain, ground, grass", 10: "sky",
                     12: "rider, cyclist"}
    encode_class_prompts(dataset, class_prompts, encoder_padding)

    # Encode dataset name
    if add_dataset_name:
        encode_dataset_name(dataset, encoder_padding)


def panoptic_cs_background_prompts_get(dataset: BaseDataset, id: int, item: Dict[str, Any]) -> Dict[str, Any]:
    assert instance_prompt_path is None
    img_name, panoptic_img, instance_ids, class_ids, crop_h, crop_w = (
        load_panoptic_image(dataset, id, item, panoptic_map_dir))

    # Get text encodings for classes/instances in panoptic map
    encoder_hidden_states = []
    unique_ids = th.unique(instance_ids)
    encoder_attention_mask = [(instance_ids == id) for id in unique_ids if id != 0]
    instance_classes = [(class_ids * mask).max() for mask in encoder_attention_mask]
    new_enc_attention_mask = []
    for inst_class, attn_mask in zip(instance_classes, encoder_attention_mask):
        if int(inst_class.item()) not in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
            continue
        encoder_hidden_state = dataset.class_prompts[int(inst_class.item())][1]
        encoder_hidden_states.append(encoder_hidden_state)
        new_enc_attention_mask.append(attn_mask)

    encoder_attention_mask = new_enc_attention_mask

    # Add dataset name
    if add_dataset_name or len(encoder_attention_mask) == 0:
        encoder_attention_mask, encoder_hidden_states = add_dataset_to_prompt(
            dataset,
            encoder_attention_mask,
            encoder_hidden_states
        )

    # Add global prompts
    if global_prompts:
        assert len(dataset.prompts) > 0, "No prompts encoded, provide a prompts file to the dataset."
        encoder_hidden_states.insert(0, dataset.prompts[img_name][1])
        encoder_attention_mask.insert(0, th.ones_like(encoder_attention_mask[0]))

    # Prepare masks and text encodings
    attention_mask_dict, text_encoding_dict = prepare_cross_attn_mask_and_text_encodings(
        encoder_attention_mask,
        encoder_hidden_states,
        crop_h,
        crop_w,
        resize_mode,
        pool_threshold,
    )

    item["cross_attn_prompts"] = CollatableItem(item=text_encoding_dict)
    item["cross_attn_mask"] = CollatableItem(item=attention_mask_dict)
    item["panoptic_img"] = panoptic_img

    return item


FEATURE_LIST = {
    "panoptic_cs_things_prompts": (panoptic_cs_things_prompts_get, panoptic_cs_things_prompts_init),
    "panoptic_cs_all_prompts": (panoptic_cs_all_prompts_get, panoptic_cs_all_prompts_init),
    "panoptic_cs_background_prompts": (panoptic_cs_background_prompts_get, panoptic_cs_all_prompts_init)
}


def load_cross_attention_data(
        datamodule: LightningDataModule,
        feature_name: str,
        enc_pad: bool,
        add_ds_name: bool,
        res_mode: str,
        glob_prompts: bool,
        pan_map_dir: str,
        pool_thresh: float = 0.5
    ) -> None:
    """Load and prepare the data required for Masked Attention and add it to all datasets in the DataModule.

    :param datamodule: PyTorch Lightning data module.
    :param feature_name: The type of data to load. Refer to FEATURE_LIST.
    :param enc_pad: If prompts should be encoded with our without encoder padding. False is recommended.
    :param add_ds_name: If the dataset name should be added as part of the prompts.
    :param res_mode: The resizing mode. Either "nearest" or "avg_pool".
    :param glob_prompts: If true, load the global prompts.
    :param pan_map_dir: The directory where the panoptic maps are saved.
    :param pool_thresh: Threshold for average pool resizing.
    """
    global encoder_padding
    encoder_padding = enc_pad
    global add_dataset_name
    add_dataset_name = add_ds_name
    global resize_mode
    resize_mode = res_mode
    global global_prompts
    global_prompts = glob_prompts
    global panoptic_map_dir
    panoptic_map_dir = pan_map_dir
    global pool_threshold
    pool_threshold = pool_thresh

    try:
        if feature_name not in FEATURE_LIST:
            raise NotImplementedError(f"Dataset feature {feature_name} does not exist!")

        for dataset in (datamodule.train_dataset, datamodule.val_dataset, datamodule.test_dataset):
            if isinstance(dataset, Subset):
                dataset = dataset.dataset

            dataset.add_load_additional(*FEATURE_LIST[feature_name])
    except AttributeError:
        raise AttributeError(f"Cannot add dataset feature {feature_name} to dataset {dataset} as"
                             f"it does not support adding additional dataset features!")
