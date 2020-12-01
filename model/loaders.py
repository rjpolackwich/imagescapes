import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

import pytorch_lightning as pl

from PIL import Image
import pandas as pd


class SkywayDataset(Dataset):

    def __init__(self, csv_file, img_dir, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transforms

#     @property
#     def transforms(self):
#         return self._transforms

#     @transforms.setter
#     def transforms(self, target_transforms):
#         self._transforms = target_transforms

    def __getitem__(self, idx):
        d = self.df.iloc[idx]
        image = Image.open(os.path.join(self.img_dir, d.Feature)).convert("RGB")
        label = torch.tensor(d[1:].tolist(), dtype = torch.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.df)


class SkywayDataMolule(pl.LightningDataModule):

    skyway_transforms = {
        'training': train_transform,
        'validation': eval_transform,
        'testing': eval_transform
    }

    def __init__(self,
                 labels_path: str = LABELS_PATH,
                 images_dir: str = IMAGES_DIR,
                 batch_size: int = BATCH_SIZE,
                 val_percent: float = 0.2,
                 test_percent: float = 0.1
                 ):
        super().__init__()
        self.labels_path = labels_path
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.val_percent = val_percent
        self.test_percent = test_percent

    def setup(self, stage=None):
        skyway_dataset = SkywayDataset(self.labels_path, self.images_dir, eval_transform)

        testing_size = int(len(skyway_dataset) * self.test_percent)
        validation_size = int(len(skyway_dataset) * self.val_percent)
        training_size = len(skyway_dataset) - (validation_size + testing_size)
        train_set, val_set, test_set = random_split(skyway_dataset,
                                                    [training_size,
                                                     validation_size,
                                                     testing_size]
                                                   )
        self.training_set = train_set
        self.validation_set = val_set
        self.testing_set = test_set

        # Assign appropriate transforms here
        # self.training_set.transforms = self.skyway_transforms['training']
        # self.validation_set.transforms = self.skyway_transforms['validation']
        # self.testing_set.transforms = self.skyway_transforms['testing']

    def train_dataloader(self):
        loader = DataLoader(self.training_set,
                            shuffle=True,
                            batch_size=self.batch_size)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.validation_set,
                            shuffle=False,
                            batch_size=self.batch_size)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.testing_set,
                            shuffle=False,
                            batch_size=self.batch_size)
        return loader


