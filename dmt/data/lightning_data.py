import random
from typing import Any, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from dmt.data.utils import extend_default_collate_fn

# Extends PyTorch's collate function by a new special class that supports collating any item to a List.
extend_default_collate_fn()


class LitDataModule(LightningDataModule):
    """The PyTorch Lightning Data Module. Responsible for providing the Trainer with the dataloaders during training."""

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        num_val: Optional[int] = None,
        num_stable_train: int = 250,
        batch_size: int = 128,
        num_workers: int = 16,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        *args, **kwargs
    ) -> None:
        """Init a LitDataModule

        :param train_dataset: A PyTorch Dataset containing training data
        :param val_dataset: A PyTorch Dataset containing training data
        :param test_dataset: A PyTorch Dataset containing training data
        :param num_val: Number of images used for calculating validation loss.
        :param num_stable_train: Number of images used to calculate a stable training loss.
        :param batch_size: Batch size of the dataloader (per GPU).
        :param num_workers: Number of workers in the dataloader (per GPU).
        :param pin_memory: pin memory as in PyTorch Dataloader.
        :param persistent_workers: persistent workers as in PyTorch Dataloader.
        """
        super().__init__()

        self.train_dataset = train_dataset
        if num_val is not None:
            indices = list(range(len(val_dataset)))
            random.shuffle(indices)
            self.val_dataset = Subset(val_dataset, indices[0:num_val])
            self.full_val_dataset = val_dataset
        else:
            self.val_dataset = val_dataset
            self.full_val_dataset = val_dataset

        self.test_dataset = test_dataset

        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        self.stable_train_dataset = Subset(train_dataset, indices[0:num_stable_train])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def train_dataloader(
            self,
            shuffle: bool = True,
            batch_size: Optional[int] = None,
    ) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size if batch_size is None else batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
            drop_last=True,
        )

    def val_dataloader(
            self,
            shuffle: bool = False,
            batch_size: Optional[int] = None,
            full_dataset: bool = False,
    ) -> DataLoader[Any]:
        """Create and return the val dataloader.

        :return: The val dataloader.
        """
        return DataLoader(
            dataset=self.val_dataset if not full_dataset else self.full_val_dataset,
            batch_size=self.batch_size if batch_size is None else batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=shuffle,
            drop_last=True,
        )

    def test_dataloader(
            self,
            shuffle: bool = False,
            batch_size: Optional[int] = None,
    ) -> Optional[DataLoader[Any]]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if self.test_dataset is not None:
            return DataLoader(
                dataset=self.test_dataset,
                batch_size=self.batch_size if batch_size is None else batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                shuffle=shuffle,
                drop_last=True,
            )
        else:
            return None

    def stable_train_dataloader(self) -> DataLoader[Any]:
        """Create and return a dataloader for stable train validation.

        :return: The stable train dataloader.
        """
        return DataLoader(
            dataset=self.stable_train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,
            drop_last=True,
        )
