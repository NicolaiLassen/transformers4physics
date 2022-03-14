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
from vit_pytorch.twins_svt import TwinsSVT


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    lr = 1e-3
    epochs = 200
    embed = 32

    data = np.load("./data/cube36_2d.npy").squeeze()
    data = data.swapaxes(1,2).reshape(500,3,36,36)
    data = data.swapaxes(2,3)
    print(data.shape)

    model = TwinsSVT(
        num_classes = 1000,       # number of output classes
        s1_emb_dim = 64,          # stage 1 - patch embedding projected dimension
        s1_patch_size = 3,        # stage 1 - patch size for patch embedding
        s1_local_patch_size = 3,  # stage 1 - patch size for local attention
        s1_global_k = 3,          # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
        s1_depth = 1,             # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
        s2_emb_dim = 128,         # stage 2 (same as above)
        s2_patch_size = 2,
        s2_local_patch_size = 3,
        s2_global_k = 3,
        s2_depth = 1,
        s3_emb_dim = 256,         # stage 3 (same as above)
        s3_patch_size = 1,
        s3_local_patch_size = 1,
        s3_global_k = 1,
        s3_depth = 5,
        s4_emb_dim = 512,         # stage 4 (same as above)
        s4_patch_size = 2,
        s4_local_patch_size = 1,
        s4_global_k = 1,
        s4_depth = 4,
        peg_kernel_size = 3,      # positional encoding generator kernel size
        dropout = 0.              # dropout
    )

    td = torch.from_numpy(data).float()

    model(td)

    exit(0)

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