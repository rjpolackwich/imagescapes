from base_model import BaseModel
from torch import nn
import torch.nn.functional as F
from torchvision import models


class ResnetClassifier(BaseModel):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.resnet(x)

    @staticmethod
    def loss(*args):
        return F.cross_entropy(*args)
