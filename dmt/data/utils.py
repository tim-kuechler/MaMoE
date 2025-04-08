from dataclasses import dataclass
from typing import List, Set, Any

from torch.utils import data

@dataclass
class CollatableItem:
    """This class allows any type of data to be collated during data loading.

    Data will be collated as list. In our case we use this e.g. if any item in batch has a tensor of different shape.
    These tensors cannot be collated by PyTorch by default so we need to wrap them in this class.
    """
    item: Any

def collate_item(batch, collate_fn_map) -> List[Set[Any]]:
    return [col_item.item for col_item in batch]

def extend_default_collate_fn():
    data._utils.collate.default_collate_fn_map.update({CollatableItem: collate_item})
