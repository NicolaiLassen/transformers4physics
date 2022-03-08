"""
=====
Training embedding model for the Lorenz numerical example.
This is a built-in model from the paper.
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""


import argparse
import logging

import torch
import h5py

from torch.optim.lr_scheduler import ExponentialLR

from config.config_phys import PhysConfig
from data_utils.enn_data_handler import LorenzDataHandler
from magtense_micro_test import MicroMagnetEmbeddingTrainer
import numpy as np

from embeddings.enn_trainer import EmbeddingTrainer


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    lr = 1e-3
    epochs = 300
    embed = 128

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Torch device: {}".format(device))

    # Set up data-loaders
    data_handler = LorenzDataHandler()
    training_loader = data_handler.createTrainingLoader(
        batch_size=64,
        block_size=4,
        stride=4,
        ndata=-1,
        file_path='./magtense_micro_test/cube36_3d.h5',
    )
    testing_loader = data_handler.createTestingLoader(
        batch_size=16,
        block_size=32,
        ndata=6,
        file_path='./magtense_micro_test/cube36_3d.h5',
    )

    # Set up model
    cfg = PhysConfig(
        n_ctx=16,
        n_embd=embed,
        n_layer=4,
        n_head=4,
        state_dims=[3],
        activation_function="gelu_new",
        initializer_range=0.05,
    )
    model = MicroMagnetEmbeddingTrainer(
        config=cfg,
    ).to(device)

    mu, std = data_handler.norm_params
    model.embedding_model.mu = mu.to(device)
    model.embedding_model.std = std.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = ExponentialLR(optimizer, gamma=0.995)

    args = argparse.ArgumentParser
    args.device = device
    args.epoch_start = 0
    args.seed = 2
    args.epochs = epochs
    args.save_steps = 25
    args.ckpt_dir = './checkpoints/magtense_micro_test/'

    trainer = EmbeddingTrainer(model, args, (optimizer, scheduler))
    
    trainer.train(training_loader, testing_loader)