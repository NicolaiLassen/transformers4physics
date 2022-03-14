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
import ast
import configparser
import logging
import sys

import torch
import h5py

from torch.optim.lr_scheduler import ExponentialLR

from config.config_phys import PhysConfig
from data_utils.enn_data_handler import LorenzDataHandler
import numpy as np
import argparse

from embeddings.enn_trainer import EmbeddingTrainer
from embeddings.magtense_micro_test_embedding import MicroMagnetEmbeddingTrainer

def read_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return (config)


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config_path = vars(args)['config']
    assert config_path is not None, 'Please provide a config file via --config'
    cfg = read_config(config_path)
    cfg_embed = cfg['EMBED']

    device = torch.device("cuda:0" if cfg['META']['device'] == 'cuda' else "cpu")
    logger.info("Torch device: {}".format(device))

    # Set up data-loaders
    data_handler = LorenzDataHandler()
    training_loader = data_handler.createTrainingLoader(
        batch_size=cfg_embed.getint('batch_size_train'),
        block_size=cfg_embed.getint('block_size_train'),
        stride=cfg_embed.getint('stride_train'),
        ndata=cfg_embed.getint('ndata_train'),
        file_path=cfg['META']['h5_file_train'],
    )
    testing_loader = data_handler.createTestingLoader(
        batch_size=cfg_embed.getint('batch_size_eval'),
        block_size=cfg_embed.getint('block_size_eval'),
        ndata=cfg_embed.getint('ndata_eval'),
        file_path=cfg['META']['h5_file_eval'],
    )

    # Set up model
    modelcfg = PhysConfig(
        n_ctx=cfg['TRANSFORMER'].getint('n_ctx'),
        n_layer=cfg['TRANSFORMER'].getint('n_layer'),
        n_embd=cfg['META'].getint('n_embd'),
        n_head=cfg['TRANSFORMER'].getint('n_head'),
        state_dims=ast.literal_eval(cfg['TRANSFORMER']['state_dims']),
        activation_function=cfg['TRANSFORMER']['activation_function'],
        initializer_range=cfg['TRANSFORMER'].getfloat('initializer_range'),
    )
    model = MicroMagnetEmbeddingTrainer(
        config=modelcfg,
    ).to(device)

    mu, std = data_handler.norm_params
    model.embedding_model.mu = mu.to(device)
    model.embedding_model.std = std.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg_embed.getfloat('lr'), weight_decay=1e-8)
    scheduler = ExponentialLR(optimizer, gamma=0.995)

    args = argparse.ArgumentParser
    args.device = device
    args.src_device = device
    args.exp_dir = cfg['META']['exp_dir'] + '/embed/'
    args.epoch_start = cfg_embed.getint('epoch_start')
    args.seed = cfg_embed.getint('seed')
    args.epochs = cfg_embed.getint('epochs')
    args.save_steps = cfg_embed.getint('save_steps')
    args.ckpt_dir = cfg['META']['checkpoint_dir']

    trainer = EmbeddingTrainer(model, args, (optimizer, scheduler))

    trainer.train(training_loader, testing_loader)
