from abc import ABC
from coco_dataset import CocoClassificationDataset
import os
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms


chip_size = 256

eval_transform = transforms.Compose([
    transforms.Resize(chip_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize(chip_size),
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class BaseModel(pl.LightningModule, ABC):
    '''
    Base class to use for ship classification.
    It

    * stubs the basic lightning classes (train_step, etc.) so only the model need be implemented elsewhere
    * sets up dataloaders that read from jsonl chip record files and automatically download and cache chips, applying user-defined transforms
    * configures an SGD optimizer and cross entropy loss
    '''
    transforms = {
        'train': train_transform,
        'val': eval_transform,
        'test': eval_transform,
    }

    @staticmethod
    def loss(*args):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {
            'loss': loss,
            'log': {'training/loss': loss},
        }

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        accuracy = (y == y_hat).float().mean()
        return {'val_loss': self.loss(y_hat, y), 'accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        tensorboard_logs = {
            'validation/accuracy': accuracy,
            'validation/loss': avg_loss,
        }
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': self.loss(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'avg_test_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=self.hparams.momentum)

    def dataloader(self, phase):
        coco_path = os.path.join(self.hparams.data_path, f'{phase}.json')
        dataset = CocoClassificationDataset(coco_path, transform=self.transforms[phase])
        shuffle = phase == 'train'
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=shuffle, num_workers=4)

    @pl.data_loader
    def train_dataloader(self):
        return self.dataloader('train')

    @pl.data_loader
    def val_dataloader(self):
        return self.dataloader('val')

    @pl.data_loader
    def test_dataloader(self):
        return self.dataloader('test')

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--learning_rate', type=float, default=0.01)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--data_path', type=str, required=True)
        return parser
