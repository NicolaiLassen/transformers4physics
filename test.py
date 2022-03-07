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
from embeddings.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding, LandauLifshitzGilbertEmbeddingTrainer
from embeddings.embedding_model import EmbeddingTrainingHead
import numpy as np

from embeddings.enn_trainer import EmbeddingTrainer


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    lr = 1e-3
    epochs = 200
    embed = 32

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
        batch_size=16,
        block_size=64,
        stride=64,
        ndata=1,
        file_path='./tests/koopman_git_2/magnet_data_train.h5',
    )
    testing_loader = data_handler.createTestingLoader(
        batch_size=1,
        block_size=64,
        ndata=4,
        file_path='./tests/koopman_git_2/magnet_data_train.h5',
    )

    # Set up model
    cfg = PhysConfig(
        n_ctx=64,
        n_embd=embed,
        n_layer=4,
        n_head=4,
        state_dims=[3],
        activation_function="gelu_new",
        initializer_range=0.05,
    )
    model = LandauLifshitzGilbertEmbeddingTrainer(
        config=cfg,
    ).to(device)

    mu, std = data_handler.norm_params
    hf = h5py.File('./tests/koopman_git_2/magnet_norm_params.h5', 'w')
    hf.create_dataset('dataset_1', data=np.array([mu.detach().cpu().numpy(),std.detach().cpu().numpy()]))
    hf.close()
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
    args.ckpt_dir = './checkpoints/mag_model_koop/'

    trainer = EmbeddingTrainer(model, args, (optimizer, scheduler))
    
    trainer.train(training_loader, testing_loader)