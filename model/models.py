import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

from torch.optim import lr_scheduler

import pytorch_lightning as pl
from pytorch_lightning.metrics.classification import Accuracy, FBeta
from pytorch_lightning import Trainer, seed_everything


class SkywayMuliLabelClassifier(pl.LightningModule):
    """
    A pytorch_lightning.LightningModule subclass. Used in
    conjunction with SkywayDataset dataloader class for training,
    validating and testing multi-label, multi-class skyway data.
    """

    def __init__(self,
                 n_labels: int = 13,
                 cnn_backbone: nn.Module = models.resnet50,
                 loss_fn: nn.Module = nn.BCEWithLogitsLoss(),
                 discriminator_optimizer = torch.optim.Adam,
                 learning_rate: float = 1e-3,
                 discriminator_scheduler = lr_scheduler.CosineAnnealingLR,
                 binary_threshold = 0.5,
                ):
        super().__init__()

        self.n_labels = n_labels
        self.loss = loss_fn
        self.optimizer = discriminator_optimizer
        self.learning_rate = learning_rate
        self.scheduler = discriminator_scheduler
        self.binary_threshold = binary_threshold
        self.cnn = cnn_backbone(pretrained=True)
        # Freeze backbone
        for param in self.cnn.parameters():
            param.requires_grad_(False)

        self.cnn.fc = nn.Sequential(
            nn.Linear(self.cnn.fc.in_features, self.cnn.fc.in_features // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.cnn.fc.in_features // 2),
            nn.Dropout(0.2),
            nn.Linear(self.cnn.fc.in_features // 2, self.cnn.fc.in_features // 4),
            nn.ReLU(),
            nn.BatchNorm1d(self.cnn.fc.in_features // 4),
            nn.Dropout(0.2),
            nn.Linear(self.cnn.fc.in_features // 4, self.n_labels)
        )

        # Set up metrics
        self.accuracy = Accuracy(threshold=self.binary_threshold,
                                 compute_on_step=True,
                                 dist_sync_on_step=True,
                                )
        self.f2 = FBeta(self.n_labels,
                        beta=2.0,
                        threshold=self.binary_threshold,
                        multilabel=True,
                        compute_on_step=True,
                        dist_sync_on_step=True,
                       )

    def forward(self, x):
        return self.cnn(x)

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
        f2 = self.f2(logits, targets)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True)
        self.log('train_f2', f2, prog_bar=True)

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
        f2 = self.f2(logits, targets)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        self.log('val_f2', f2, prog_bar=True)

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
        f2 = self.f2(logits, targets)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', accuracy, prog_bar=True)
        self.log('test_f2', f2, prog_bar=True)

        return loss

    def configure_optimizers(self):
        disc_optimizer = self.optimizer(self.cnn.parameters(), lr=self.learning_rate)
        disc_scheduler = self.scheduler(disc_optimizer, T_max=5, eta_min=0.005)
        return {
            "optimizer": disc_optimizer,
            "lr_scheduler": disc_scheduler
        }

