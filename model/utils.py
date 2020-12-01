import os
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import skimage.io as skio
import pandas as pd
import seaborn as sns


def load_sparse_encodings(csv_path, **kwargs):
    df = pd.read_csv(csv_path, **kwargs)
    return df


def display_label_distribution(df, figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    df.iloc[:, 1:].sum(axis=0).plot.pie(autopct='%1.1f%%', shadow=True, startangle=90, ax=ax)
    ax.axis("equal")
    plt.show()


def display_label_corrs(df):
    sns.headmap(df.iloc[:, 1:].corr(), cmap="RdYlBu", vmin=-1, vmax=1)


def visualize_image(df, idx, img_path="/home/ubuntu/data/chips", figsize=(10, 10)):
    classes = df.columns[1:].tolist()
    fd = df.iloc[idx]
    image = fd.Feature
    label = fd[1:].tolist()

    image = Image.open(os.path.join(img_path, image))
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.grid(False)
    classes = np.array(classes)[np.array(label, dtype=np.bool)]
    for i, s in enumerate(classes):
        ax.text(0, i*20, s, verticalalignment='top', color='white', fontsize=16, weight='bold')
    plt.show()


