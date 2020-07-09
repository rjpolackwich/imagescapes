#!/usr/bin/env python
from argparse import ArgumentParser
import logging
from models import ResnetClassifier
import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger
import torch


logging.getLogger("botocore").setLevel(logging.CRITICAL)

pl.seed_everything(777)
model_map = {
    'resnet': ResnetClassifier,
}


def save_model(model, fname):
    batch_size = model.hparams.batch_size
    sample_input = torch.randn(batch_size, 3, 256, 256, device='cuda:0')
    model.to('cuda:0')
    torch.onnx.export(model, sample_input, fname, input_names=['input_batch'], output_names=['output_logits'])


def main():
    # parse args to get model
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', help='only resnet is supported at this time')
    temp_args, _ = parser.parse_known_args()
    if temp_args.model not in model_map.keys():
        raise ValueError('Invalid model; choose resnet')
    model_cls = model_map[temp_args.model]

    # parse model args
    parser = model_cls.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    model = model_cls(hparams=args)

    # Set up logger & trainer
    logger = TestTubeLogger(name="default", debug=False, save_dir="model_logs/")
    trainer = pl.Trainer.from_argparse_args(args, logger=logger)

    # train and save model
    trainer.fit(model)
    save_model(model, 'model.onnx')


if __name__ == "__main__":
    main()
