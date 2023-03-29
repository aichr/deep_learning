import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):

    def __init__(
            self, data_dir: str, batch_size: int, num_workers: int,
            shuffle: bool = True, pin_memory: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def prepare_data(self):
        is_download = True
        if os.path.exists('data/MNIST'):
            is_download = False

        # Download the MNIST dataset if it doesn't already exist
        datasets.MNIST(self.data_dir, train=True, download=is_download)
        datasets.MNIST(self.data_dir, train=False, download=is_download)

    def setup(self, stage=None):
        # Define the transforms to be applied to the data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load the MNIST dataset
        if stage == 'fit' or stage is None:
            self.train_dataset = datasets.MNIST(
                self.data_dir, train=True, transform=transform)
            self.val_dataset = datasets.MNIST(
                self.data_dir, train=False, transform=transform)
        if stage == 'test' or stage is None:
            self.test_dataset = datasets.MNIST(
                self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,      # turn off shuffle for validation
            num_workers=self.num_workers,
            pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,      # turn off shuffle for test
            num_workers=self.num_workers,
            pin_memory=self.pin_memory)
