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
import sys
import logging
import h5py
from turtle import st
import torch
from torch.optim.lr_scheduler import ExponentialLR
from embedding_lorenz import LorenzEmbedding, LorenzEmbeddingTrainer
from enn_data_handler import LorenzDataHandler

from viz_lorenz import LorenzViz
from enn_trainer import *
from lorenz_data import *
from config_phys import PhysConfig

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    lr = 1e-3
    epochs = 50

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    if(torch.cuda.is_available()):
        use_cuda = "cuda"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Torch device: {}".format(device))

    # Load transformer config file

    # Set up data-loaders
    data_handler = LorenzDataHandler()
    training_loader = data_handler.createTrainingLoader(
        batch_size=512,
        block_size=16,
        file_path='./lorenz_data_train.h5',
    )
    testing_loader = data_handler.createTestingLoader(
        batch_size=8,
        block_size=32,
        file_path='./lorenz_data_test.h5',
    )

    # Set up model
    cfg = PhysConfig(
        n_ctx=64,
        n_embd=32,
        n_layer=4,
        n_head=4,
        state_dims=[3],
        activation_function="gelu_new",
        initializer_range=0.05,
    )
    model = LorenzEmbeddingTrainer(
        config=cfg,
    ).to(device)
    model.embedding_model.koopmanOperation(torch.rand(1,32).cuda())
    print(model.embedding_model.koopmanOperator)

    mu, std = data_handler.norm_params
    hf = h5py.File('lorenz_norm_params.h5', 'w')
    hf.create_dataset('dataset_1', data=np.array([mu.detach().cpu().numpy(),std.detach().cpu().numpy()]))
    hf.close()
    model.mu = mu.to(device)
    model.std = std.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = ExponentialLR(optimizer, gamma=0.995)

    args = argparse.ArgumentParser
    args.device = device
    args.epoch_start = 0
    args.seed = 2
    args.epochs = epochs
    args.save_steps = 25
    args.ckpt_dir = './tests/koopman_git_2/koop_model/'

    trainer = EmbeddingTrainer(
        model, args, (optimizer, scheduler))
    trainer.train(training_loader, testing_loader)
    print(model.embedding_model.koopmanOperator)
    hf = h5py.File('koopman_op.h5', 'w')
    hf.create_dataset('dataset_1', data=model.embedding_model.koopmanOperator.detach().cpu().numpy())
    hf.close()