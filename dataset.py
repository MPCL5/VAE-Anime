from pytorch_lightning import LightningDataModule
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from typing import List, Optional, Sequence, Union, Any, Callable


IMG_SIZE = 64  # input dimension
DATA_DIR = './data'
TEST_SIZE = 1000
VALIDATION_SIZE = 1000


def get_img_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_img_transform(self):
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.CenterCrop(148),
            T.Resize(self.patch_size, antialias=True),
            T.ToTensor(),
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = ImageFolder(
            root=DATA_DIR, transform=self.get_img_transform())

        self.val_dataset = torch.utils.data.Subset(
            dataset, range(VALIDATION_SIZE))
        self.train_dataset = torch.utils.data.Subset(
            dataset, range(VALIDATION_SIZE, len(dataset)))

        print(len(self.train_dataset))
        print(len(self.val_dataset))


#       ===============================================================


    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
