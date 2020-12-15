import os
import argparse
from pathlib import Path
from typing import Optional, Union, Generator

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn import Module
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import Accuracy, FBeta, Recall
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint, GPUStatsMonitor

from PIL import Image
import pandas as pd


BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)

def _make_trainable(module: Module) -> None:
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module: Module,
                      train_bn: bool = True) -> None:
    """Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable
            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module=child, train_bn=train_bn)


def freeze(module: Module,
           n: Optional[int] = None,
           train_bn: bool = True) -> None:
    """Freezes the layers up to index n (if n is not None).
    Args:
        module: The module to freeze (at least partially)
        n: Max depth at which we stop freezing the layers. If None, all
            the layers of the given module will be frozen.
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    n_max = len(children) if n is None else int(n)

    for child in children[:n_max]:
        _recursive_freeze(module=child, train_bn=train_bn)

    for child in children[n_max:]:
        _make_trainable(module=child)



def filter_params(module: Module,
                  train_bn: bool = True) -> Generator:
    """Yields the trainable parameters of a given module.
    Args:
        module: A given module
        train_bn: If True, leave the BatchNorm layers in training mode
    Returns:
        Generator
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                if param.requires_grad:
                    yield param
    else:
        for child in children:
            for param in filter_params(module=child, train_bn=train_bn):
                yield param


def _unfreeze_and_add_param_group(module: Module,
                                  optimizer: Optimizer,
                                  lr: Optional[float] = None,
                                  train_bn: bool = True):
    """Unfreezes a module and adds its parameters to an optimizer."""
    _make_trainable(module)
    params_lr = optimizer.param_groups[0]['lr'] if lr is None else float(lr)
    optimizer.add_param_group(
        {'params': filter_params(module=module, train_bn=train_bn),
         'lr': params_lr / 10.,
         })


class SkywayMultiLabelClassifier(pl.LightningModule):
    """
    A pytorch_lightning.LightningModule subclass. Used in
    conjunction with SkywayDataset dataloader class for training,
    validating and testing multi-label, multi-class skyway data.
    """

    def __init__(
            self,
            labels_path: Union[str, Path] = LABELS_PATH,
            images_dir: Union[str, Path] = IMAGES_DIR,
            batch_size: int = BATCH_SIZE,
            val_percent: float = 0.25,
            test_percent: float = 0.05,
            num_workers: int = 16,
            num_labels: int = 8,
            backbone: str = 'resnet50',
            train_bn: bool = True,
            milestones: tuple = (5, 10),
            loss_fn: Module = nn.BCEWithLogitsLoss(),
            optimizer: str = 'AdamW',
            learning_rate: float = 1e-3,
            scheduler = lr_scheduler.MultiStepLR,
            lr_scheduler_gamma: float = 1e-1,
            binary_threshold: float = 0.5,
            **kwargs,
            ) -> None:

        super().__init__()

        self.labels_path = labels_path
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.val_percent = val_percent
        self.test_percent = test_percent
        self.num_workers = num_workers
        self.num_labels = num_labels
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.loss = loss_fn
        self.optimizer = getattr(torch.optim, optimizer)
        self.learning_rate = learning_rate
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.scheduler = scheduler
        self.binary_threshold = binary_threshold

        self._build_model()
        self._build_metrics()
        self.save_hyperparameters()

    def _build_model(self):
        model_cls = getattr(models, self.backbone)
        backbone = model_cls(pretrained=True)

        fe_layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*fe_layers)
        freeze(module=self.feature_extractor, train_bn=self.train_bn)

        self.fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, backbone.fc.in_features // 2),
            nn.ReLU(),
            nn.BatchNorm1d(backbone.fc.in_features // 2),
            nn.Dropout(0.2),
            nn.Linear(backbone.fc.in_features // 2, backbone.fc.in_features // 4),
            nn.ReLU(),
            nn.BatchNorm1d(backbone.fc.in_features // 4),
            nn.Dropout(0.2),
            nn.Linear(backbone.fc.in_features // 4, self.num_labels)
        )

    def _build_metrics(self):
        self.accuracy = Accuracy(threshold=self.binary_threshold,
                                 compute_on_step=True,
                                 dist_sync_on_step=True,
                                )

        self.recall = Recall(self.num_labels,
                             threshold=self.binary_threshold,
                             multilabel=True,
                             compute_on_step=True,
                             dist_sync_on_step=True
                            )

        self.f2 = FBeta(self.num_labels,
                        beta=2.0,
                        threshold=self.binary_threshold,
                        multilabel=True,
                        compute_on_step=True,
                        dist_sync_on_step=True,
                       )


    def train(self, mode=True):
        super().train(mode=mode)

        epoch = self.current_epoch
        if epoch < self.milestones[0] and mode:
            freeze(module=self.feature_extractor,
                    train_bn=self.train_bn)

        elif self.milestones[0] <= epoch < self.milestones[1] and mode:
            freeze(module=self.feature_extractor,
                    n=-2,
                    train_bn=self.train_bn)

    def on_epoch_start(self):
        """
        Unfreze layers progressively with this hook
        """
        optimizer = self.trainer.optimizers[0]
        if self.current_epoch == self.milestones[0]:
            _unfreeze_and_add_param_group(module=self.feature_extractor[-2:],
                    optimizer=optimizer,
                    train_bn=self.train_bn)

        elif self.current_epoch == self.milestones[1]:
            _unfreeze_and_add_param_group(module=self.feature_extractor[:-2],
                    optimizer=optimizer,
                    train_bn=self.train_bn)


    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        return {
            "y_hat": y_hat,
            "target": y
        }

    def training_step_end(self, outputs):
        y_hats = outputs['y_hat']
        targets = outputs['target']
        logits = torch.sigmoid(y_hats)

        loss = self.loss(y_hats, targets)
        accuracy = self.accuracy(logits, targets)
        recall = self.recall(logits, targets)
        f2 = self.f2(logits, targets)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', accuracy, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_recall', recall, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_f2', f2, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        return {
            "y_hat": y_hat,
            "target": y
        }

    def validation_step_end(self, outputs):
        y_hats = outputs['y_hat']
        targets = outputs['target']

        logits = torch.sigmoid(y_hats)

        loss = self.loss(y_hats, targets)
        accuracy = self.accuracy(logits, targets)
        recall = self.recall(logits, targets)
        f2 = self.f2(logits, targets)

        self.log('val_loss', loss, on_step=True,  on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', recall, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f2', f2, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        return {
            "y_hat": y_hat,
            "target": y
        }

    def test_step_end(self, outputs):
        y_hats = outputs['y_hat']
        targets = outputs['target']

        logits = torch.sigmoid(y_hats)

        loss = self.loss(y_hats, targets)
        accuracy = self.accuracy(logits, targets)
        recall = self.recall(logits, targets)
        f2 = self.f2(logits, targets)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', accuracy, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        self.log('test_f2', f2, prog_bar=True)

        return loss


    def configure_optimizers(self):
        disc_optimizer = self.optimizer(
                filter(
                    lambda p: p.requires_grad,
                    self.parameters()),
                lr=self.learning_rate
                )

#        disc_scheduler = self.scheduler(
#                disc_optimizer,
#                mode='min',
#                factor=0.1,
#                patience=4,
#                verbose=True
#                )
        disc_scheduler = self.scheduler(
                disc_optimizer,
                milestones=self.milestones,
                gamma=self.lr_scheduler_gamma
                )

#        return {
#            "optimizer": disc_optimizer,
#            "lr_scheduler": disc_scheduler,
#            "monitor": "val_loss"
#        }
        return [disc_optimizer], [disc_scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument(
                '--labels-path',
                type=str,
                default=LABELS_PATH,
                metavar='PATH',
                help='Full path to sparse encoded datapoints',
                dest='labels_path'
                )
        parser.add_argument(
                '--images-dir',
                type=str,
                default=IMAGES_DIR,
                metavar='DIR',
                help='Directory containing images',
                dest='images_dir'
                )
        parser.add_argument(
                '--batch-size',
                type=int,
                default=BATCH_SIZE,
                metavar='BS',
                help='Batch size',
                dest='batch_size'
                )
        parser.add_argument(
                '--pval',
                type=float,
                default=0.25,
                metavar='PV',
                help='Percent of dataset to use for validation',
                dest='val_percent',
                )
        parser.add_argument(
                '--ptest',
                type=float,
                default=0.05,
                metavar='PT',
                help='Percent of dataset to use for testing',
                dest='test_percent'
                )
        parser.add_argument(
                '--num-workers',
                type=int,
                default=8,
                metavar='W',
                help='Number of CPU workers',
                dest='num_workers'
                )
        parser.add_argument(
                '--backbone',
                type=str,
                default='resnet50',
                metavar='BK',
                help='Name (as in ``torchvision.models``) of the feature extractor'
                )
        parser.add_argument(
                '--train-bn',
                default=True,
                type=bool,
                metavar='TB',
                help='Whether the BatchNorm layers of the backbone should be trainable',
                dest='train_bn'
                )
        parser.add_argument(
                '--lr',
                '--learning-rate',
                type=float,
                default=1e-3,
                metavar='LR',
                help='Initial learning rate',
                dest='learning_rate'
                )
        parser.add_argument(
                '--milestones',
                type=list,
                default=[5, 10],
                metavar='M',
                help='Epoch milestones at which backbone fine-tuning proceeds',
                )
        parser.add_argument(
                '--optimizer',
                type=str,
                default='AdamW',
                metavar='O',
                help='Optimizer to use in training'
                )

        return parser


