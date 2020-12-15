#!/usr/bin/env python

from argparse import ArgumentParser
import logging

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.logging import TestTubeLogger
import torch

from transforms import build_transform
from loaders import SkywayDataModule
from models import SkywayMultiLabelClassifier


def save_model(model, fname):
    batch_size = model.hparams.batch_size
    sample_input = torch.randn(batch_size, 3, 256, 256, device='cuda:0')
    model.to('cuda:0')
    torch.onnx.export(
        model,
        sample_input,
        fname,
        input_names=['input_batch'],
        output_names=['output_logits']
    )


def main(hparams):
    seed_everything(42)

    logger = TestTubeLogger(
            name='skyway',
            debug=False,
            save_dir='/home/ubuntu/data/notebooks/model_logs'
            )

    mc_callback = ModelCheckpoint(
            monitor='val_loss',
            save_top_k=-1,
            mode='min',
            save_weights_only=False,
            period=2
            )

    lr_monitor = LearningRateMonitor(logging_interval=None)
    gpu_monitor = GPUStatsMonitor()
    callbacks = [mc_callback]

    dvars = [vdesc[0] for vdesc in SkywayDataModule.get_init_arguments_and_types()[1:]]
    dcfg = {vname: getattr(hparams, vname) for vname in dvars if hasattr(hparams, vname)}
    sdm = SkywayDataModule(**dcfg)
    sdm.setup('train')

    num_classes = len(sdm.training_set.dataset.df.columns[1:].tolist())
    mcfg = vars(hparams)
    mcfg['num_classes'] = num_classes
    model = SkywayMultiLabelClassifier(**mcfg)

    trainer = Trainer(
            gpus=-1,
            max_epochs=hparams.max_epochs,
            accelerator='ddp',
            logger=logger,
            callbacks=callbacks,
            )


    trainer.fit(model, sdm)

def get_args() -> argparse.Namespace:
    parent_parser = argparse.ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parent_parser)
    parser = SkywayMultiLabelClassifier.add_model_specific_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
