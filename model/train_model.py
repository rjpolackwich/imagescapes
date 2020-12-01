#!/usr/bin/env python

from argparse import ArgumentParser
import logging

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.logging import TestTubeLogger
import torch

from transforms import build_transform
from loaders import SkywayDataModule
from models import SkywayMultiLabelClassifier


DEFAULT_BATCH_SIZE = 32
DEFAULT_INPUT_IMAGE_SIZE = 256
DEFAULT_LABELS_PATH = "/home/ubuntu/data/sample_sparse_encoings.csv"
DEFAULT_IMAGES_DIR = "/home/ubuntu/data/chips"

seed_everything(777)


def save_model(model, fname):
    batch_size = model.hparams.batch_size
    sample_input = torch.randn(batch_size, 3, 256, 256, device='cuda:0')
    model.to('cuda:0')
    torch.onnx.export(model, sample_input, fname, input_names=['input_batch'], output_names=['output_logits'])


def main(args):
    datamod = SkywayDataModule(args.labels_path,
                               args.images_dir,
                               args.batch_size,
                               )

    datamod.setup("train")
    model = model_cls(hparams=args)

    # Set up logger & trainer
    logger = TestTubeLogger(name="default", debug=False, save_dir="model_logs/")
    trainer = Trainer.from_argparse_args(args, logger=logger)

    # train and save model
    trainer.fit(model, datamod)
    save_model(model, 'model.onnx')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = SkywayMultiLabelClassifier.add_model_specific_args(parser)
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--model-name', type=str, default='model.onnx', help='output ONNX model name')
    parser.add_argument('--chip-size', type=int, default=DEFAULT_INPUT_IMAGE_SIZE, help='resize images to this size')
    parser.add_argument('--labels-path', type=str, default=DEFAULT_LABELS_PATH, help='full path to sparse encoded labels csv file')
    parser.add_argument('--images-dir', type=str, default=DEFAULT_IMAGES_DIR, help='full path to directory of image chip jpgs')

    args = parser.parse_args()
    main(args)
