import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

import torch.optim as optim
from torch.optim import lr_scheduler

from tqdm import trange
from sklearn.metrics import precision_score, f1_score

from PIL import Image

from sklearn.preprocessing import MultiLabelBinarizer

import seaborn as sns
import pandas as pd

import numpy as np
import skimage.io as skio


eval_transform = transforms.Compose([
    transforms.Resize(chip_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize((chip_size, chip_size)),
#    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


batch_size = 32
dataset = SkywayDataset("/home/ubuntu/data/sample_sparse_encodings.csv",
                        "/home/ubuntu/data/chips",
                        transforms=train_transform)

valid_no = int(len(dataset) * 0.20)

training_set, validation_set = random_split(dataset, [len(dataset) - valid_no, valid_no])
#print(f'''training set length: {len(training_set)}, validation set length: {len(validation_set)}''')

dataloader = {"train": DataLoader(training_set, shuffle=True, batch_size=batch_size),
              "val": DataLoader(validation_set, shuffle=True, batch_size=batch_size)}

def visualize_label_dist(df):
    fig1, ax1 = plt.subplots(figsize=(10,10))
    df.iloc[:,1:].sum(axis=0).plot.pie(autopct='%1.1f%%', shadow=True, startangle=90, ax=ax1)
    ax1.axis("equal")
    plt.show()


def visualize_label_corr(df):
    sns.heatmap(df.iloc[:,1:].corr(), cmap="RdYlBu", vmin=-1, vmax=1)


def visualize_image(idx, classes=classes):
    fd = d.iloc[idx]
    image = fd.Feature
    label = fd[1:].tolist()
    print(image)

    image = Image.open("/home/ubuntu/data/chips/" + image)
    #print(image.shape)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(image)
    ax.grid(False)
    classes = np.array(classes)[np.array(label, dtype=np.bool)]
    for i, s in enumerate(classes):
        ax.text(0, i*20, s, verticalalignment='top', color='white', fontsize=16, weight='bold')
    plt.show()
    return image



class SkywayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transforms

    def __getitem__(self, idx):
        d = self.df.iloc[idx]
        image = Image.open(os.path.join(self.img_dir, d.Feature)).convert("RGB")
        label = torch.tensor(d[1:].tolist(), dtype = torch.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.df)


def create_head(num_features, num_classes, dropout_prob=0.5, activation_func=nn.ReLU):
    features_lst = [num_features, num_features//2, num_features//4]
    layers = list()
    for in_f, out_f in zip(features_lst[:-1], features_lst[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activation_func())
        layers.append(nn.BatchNorm1d(out_f))
        if dropout_prob != 0:
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(features_lst[-1], num_classes))
    return nn.Sequential(*layers)



model = models.resnet50(pretrained=True)
num_features = model.fc.in_features

def freeze_pretrained(model):
    for param in model.parameters():
        param.requires_grad_(False)
    return model


top_head = create_head(num_features, 13)
model.fc = top_head


criterion = nn.BCEWithLogitsLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# specify optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
sgdr_partial = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.005)


def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=5):
    for epoch in trange(num_epochs, desc="Epochs"):
        result = []
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                scheduler.step()
            else:
                model.eval()

            # Keep track of training, validation loss
            running_loss = 0.0
            running_corrects = 0.0

            for data, target in data_loader[phase]:
                data, target = data.to(device), target.to(device)
                print(data.shape)

                with torch.set_grad_enabled(phase=="train"):
                    # Feed input
                    output = model(data)
                    # Calculate loss
                    loss = criterion(output, target)
                    predictions = torch.sigmoid(output).data > 0.35
                    predictions = predictions.to(torch.float32)

                    if phase == "train":
                        # Backwards pass: compute gradient of the loss w.r.t. model params
                        loss.backward()
                        # Update model params
                        optimizer.step()
                        # Zero the grad to stop accumulation
                        optimizer.zero_grad()

                    running_loss += loss.item() * data.size(0)
                    running_corrects += f1_score(target.to("cpu").to(torch.int).numpy(),
                                                predictions.to("cpu").to(torch.int).numpy(),
                                                average="samples") * data.size(0)

        epoch_loss = running_loss / len(data_loader[phase].dataset)
        epoch_acc = running_corrects / len(data_loader[phase].dataset)

        result.append('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    print(result)

