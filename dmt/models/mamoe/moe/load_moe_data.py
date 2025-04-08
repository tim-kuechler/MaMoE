from typing import Any, Dict

import torch as th
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset

from dmt.data.utils import CollatableItem
from dmt.data import BaseDataset
from dmt.models.mamoe.utils import (load_panoptic_image, prepare_moe_mask)

# Yes, I know that handling this stuff via global variable is ugly and probably all the code beneath should be
# done via classes and inheritance
resize_mode = "nearest"
panoptic_map_dir = None
pool_threshold = 0.5


def panoptic_cs_extreme_get(dataset: BaseDataset, id: int, item: Dict[str, Any]) -> Dict[str, Any]:
    img_name, panoptic_img, instance_ids, class_ids, crop_h, crop_w = (
        load_panoptic_image(dataset, id, item, panoptic_map_dir))

    # Get things and stuff map
    car_mask = class_ids == 13
    person_mask = th.isin(class_ids, th.tensor([11, 12]))
    truck_mask = class_ids == 14
    bus_mask = class_ids == 15
    motor_bicycle_mask = th.isin(class_ids, th.tensor([17, 18]))
    vegetation_mask = th.isin(class_ids, th.tensor([8, 9]))
    wall_fence_mask = th.isin(class_ids, th.tensor([3, 4]))
    road_sidewalk_mask = th.isin(class_ids, th.tensor([0, 1]))
    building_mask = class_ids == 2
    rest_mask = ~th.isin(class_ids, th.tensor([13, 11, 12, 14, 15, 17, 18, 8, 9, 3, 4, 0, 1, 2]))
    mask_list = [car_mask, person_mask, truck_mask, bus_mask, motor_bicycle_mask, vegetation_mask, wall_fence_mask,
                 road_sidewalk_mask, building_mask, rest_mask]

    # Prepare masks and text encodings
    dispatch_mask_dict, combine_mask_dict = prepare_moe_mask(
        mask_list,
        crop_h,
        crop_w,
        resize_mode,
        pool_threshold,
    )

    item["moe_mask"] = CollatableItem(item=(dispatch_mask_dict, combine_mask_dict))
    item["moe_binary_mask"] = car_mask

    return item


def panoptic_cs_extreme_things_get(dataset: BaseDataset, id: int, item: Dict[str, Any]) -> Dict[str, Any]:
    img_name, panoptic_img, instance_ids, class_ids, crop_h, crop_w = (
        load_panoptic_image(dataset, id, item, panoptic_map_dir))

    # Get things and stuff map
    car_mask = class_ids == 13
    person_mask = th.isin(class_ids, th.tensor([11, 12]))
    truck_bus_mask = th.isin(class_ids, th.tensor([14, 15]))
    motor_bicycle_mask = th.isin(class_ids, th.tensor([17, 18]))
    rest_mask = ~th.isin(class_ids, th.tensor([13, 11, 12, 14, 15, 17, 18]))
    mask_list = [car_mask, person_mask, truck_bus_mask, motor_bicycle_mask, rest_mask]

    # Prepare masks and text encodings
    dispatch_mask_dict, combine_mask_dict = prepare_moe_mask(
        mask_list,
        crop_h,
        crop_w,
        resize_mode,
        pool_threshold,
    )

    item["moe_mask"] = CollatableItem(item=(dispatch_mask_dict, combine_mask_dict))
    item["moe_binary_mask"] = car_mask

    return item


def panoptic_cs_things_get(dataset: BaseDataset, id: int, item: Dict[str, Any]) -> Dict[str, Any]:
    img_name, panoptic_img, instance_ids, class_ids, crop_h, crop_w = (
        load_panoptic_image(dataset, id, item, panoptic_map_dir))

    # Get things and stuff map
    things_mask = class_ids != 19
    stuff_mask = class_ids == 19
    mask_list = [things_mask, stuff_mask]

    # Prepare masks and text encodings
    dispatch_mask_dict, combine_mask_dict = prepare_moe_mask(
        mask_list,
        crop_h,
        crop_w,
        resize_mode,
        pool_threshold,
    )

    item["moe_mask"] = CollatableItem(item=(dispatch_mask_dict, combine_mask_dict))
    item["moe_binary_mask"] = things_mask

    return item


FEATURE_LIST = {
    "panoptic_cs_things": (panoptic_cs_things_get, None),
    "panoptic_cs_extreme": (panoptic_cs_extreme_get, None),
    "panoptic_cs_extreme_things": (panoptic_cs_extreme_things_get, None),
}


def load_moe_data(
        datamodule: LightningDataModule,
        feature_name: str,
        res_mode: str,
        pan_map_dir: str,
        pool_thresh: float = 0.5,
) -> None:
    """Load and prepare the data required for MoE and add it to all datasets in the DataModule.

    :param datamodule: PyTorch Lightning data module.
    :param feature_name: The type of data to load. Refer to FEATURE_LIST.
    :param res_mode: The resizing mode. Either "nearest" or "avg_pool".
    :param pan_map_dir: The directory where the panoptic maps are saved.
    :param pool_thresh: Threshold for average pool resizing.
    """
    global resize_mode
    resize_mode = res_mode
    global panoptic_map_dir
    assert pan_map_dir is not None
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
