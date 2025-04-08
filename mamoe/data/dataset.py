import csv
import json
import re
import random
import logging
from typing import Any, Dict, List, Optional, Union, Callable
import os

from torch.utils.data import Dataset
from torchvision.io import ImageReadMode
from torchvision.transforms.v2 import (Compose, Transform, Resize, ToTensor, RandomCrop,
                                       RandomHorizontalFlip, CenterCrop)
from torchvision.transforms.v2 import functional as F
import torchvision.io as IO

from mamoe.utils import encode_prompts

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AddValue(Transform):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __call__(self, tensor):
        return tensor + self.value


class MultiplyValue(Transform):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __call__(self, tensor):
        return tensor * self.value


class DivideValue(Transform):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def __call__(self, tensor):
        return tensor / self.value


class BaseDataset(Dataset):
    def __init__(
            self,
            dataset_name: str,
            dataset_dir: str,
            split: str,
            model_name: Optional[str] = None,
            prompts_file: Optional[str] = None,
            instance_prompts_file: Optional[str] = None,
            transforms: Optional[Union[List[Transform], Compose]] = None,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.split = split
        self.model_name = model_name
        self.prompts_file = prompts_file
        self.instance_prompts_file = instance_prompts_file
        self.transforms = self.get_default_transforms() if transforms is None else transforms

        # Init image paths for dataset
        self.load_additional: List[Callable] = []
        self.prompts = {}
        self.instance_prompts = {}
        self.image_paths = []
        self.update_image_paths()

        # Encode prompts
        if prompts_file is not None:
            self.prepare_prompts()
        if instance_prompts_file is not None:
            self.prepare_instance_prompts()

    def update_image_paths(self) -> None:
        dataset_split_dir = os.path.join(self.dataset_dir, self.split)
        image_paths = []
        for img_name in sorted(os.listdir(dataset_split_dir), key=lambda x: int(re.search(r'\d+', x).group())):
            image_paths.append(os.path.join(dataset_split_dir, img_name))
        self.image_paths = image_paths

    def prepare_prompts(self) -> None:
        assert self.model_name is not None, "Model name required to encode prompts"

        # Read prompts from csv
        prompts = {}
        with open(self.prompts_file, mode='r', encoding='utf-8') as file_ref:
            csv_reader = csv.DictReader(file_ref)
            for row in csv_reader:
                prompts[row["img_name"]] = row["prompt"]
        self.prompts = prompts

        # Encode prompts
        if self.model_name is not None:
            self.prompts = encode_prompts(self.model_name, self.prompts, True)

    def prepare_instance_prompts(self):
        """Encodes instance prompts from a json file and add them to the dataset."""
        assert self.model_name is not None, "Model name required to encode prompts"

        with open(self.instance_prompts_file, 'r', encoding="utf-8") as f:
            instance_prompt_dict = json.load(f)

        # Flatten into one dict which fill speed up encoding the prompts
        flattened_instance_prompt_dict = {
            f"{outer_key}_{inner_key}": inner_value
            for outer_key, inner_dict in instance_prompt_dict.items()
            for inner_key, inner_value in inner_dict.items()
        }

        # Encode prompts
        flattened_instance_prompt_dict = encode_prompts(self.model_name, flattened_instance_prompt_dict, pad_tokens=False)

        # Reverse the fattening operation
        instance_prompt_dict = {}
        for flat_key, value in flattened_instance_prompt_dict.items():
            outer_key, inner_key = flat_key.rsplit("_", 1)
            if outer_key not in instance_prompt_dict:
                instance_prompt_dict[outer_key] = {}
            instance_prompt_dict[outer_key][int(inner_key)] = value

        self.instance_prompts = instance_prompt_dict

    @staticmethod
    def get_default_transforms() -> Compose:
        return Compose([
            ToTensor(),
            Resize(512, interpolation=F.InterpolationMode.BICUBIC),
            RandomCrop(512),
            RandomHorizontalFlip(0.5),
            DivideValue(127.5),
            AddValue(-1.0),
        ])

    def add_load_additional(self, get: Callable, init: Optional[Callable] = None) -> None:
        """Adds additional data that is loaded in __get_item__.

        :param get: This function is called in __get_item__ to load additional data. __get_item__ passes the item dict
            with all currently loaded stuff (image, image_name, encoded prompt, ...) and expects to get the same
            dictionary back with the additional data added to the dictionary. In theory, it is also fine to edit
            or replace the already loaded data by overwriting its dictionary key.
        :param init: This optional function is called directly here and does some initial data initialization
            (e.g. prompt encoding, ...) if required for the data that is to be loaded later in __get_item__.
        """
        if init is not None:
            init(self)
        self.load_additional.append(get)

    def __getitem__(self, img_id: int) -> Dict[str, Any]:
        item = {}
        image_path = self.image_paths[img_id]
        item["img_path"] = image_path

        # Load images and features
        img = IO.read_image(image_path, mode=ImageReadMode.RGB)

        # Transform image
        transforms = self.transforms if isinstance(self.transforms, List) else self.transforms.transforms
        for transform in transforms:
            if isinstance(transform, Resize):
                img = transform(img)
                item["orig_size"] = (img.shape[-2], img.shape[-1])
            elif isinstance(transform, RandomCrop):
                top, left, crop_h, crop_w = RandomCrop.get_params(img, output_size=transform.size)
                img = F.crop(img, top, left, crop_h, crop_w)
                item["cropped_size"] = tuple(transform.size)
                item["crop_coords_top_left"] = (top, left)
            elif isinstance(transform, CenterCrop):
                h, w = img.shape[1], img.shape[2]
                img = transform(img)
                item["cropped_size"] = tuple(transform.size)
                item["crop_coords_top_left"] = (((h - transform.size[0]) // 2), ((w - transform.size[1]) // 2))
            elif isinstance(transform, RandomHorizontalFlip):
                if random.random() < transform.p:
                    img = F.horizontal_flip(img)
                    item["flipped"] = True
                else:
                    item["flipped"] = False
            else:
                img = transform(img)

        item["pixel_values"] = img

        # Add missing image information
        if "orig_size" not in item:
            item["orig_size"] = (img.shape[-2], img.shape[-1])
        if "cropped_size" not in item:
            item["cropped_size"] = item["orig_size"]
        if "crop_coords_top_left" not in item:
            item["crop_coords_top_left"] = (0, 0)

        # Add prompt
        if self.prompts_file is not None and self.model_name is not None:
            item["prompt"] = self.prompts[os.path.basename(image_path)][0]
            # Embeddings are only computed if a model is given
            if self.model_name is not None:
                item["prompt_emb"] = self.prompts[os.path.basename(image_path)][1]
                if len(self.prompts[os.path.basename(image_path)]) == 3:
                    item["prompt_emb_2"] = self.prompts[os.path.basename(image_path)][2]

        # Add externally added additional features
        for get_feature in self.load_additional:
            item = get_feature(self, img_id, item)

        return item

    def __len__(self):
        return len(self.image_paths)
