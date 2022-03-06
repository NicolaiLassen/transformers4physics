"""
=====
Training transformer model for the Lorenz numerical example.
This is a built-in model from the paper.
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import argparse
import sys
import logging
import torch
from config.config_phys import PhysConfig
from data.enn_data_handler import LorenzDataHandler
from config.args import TrainingArguments
from transformer.phys_transformer_gpt2 import PhysformerGPT2
from transformer.phys_transformer_helpers import PhysformerTrain
from embedding.embedding_lorenz import LorenzEmbedding
from viz.viz_lorenz import LorenzViz
from data.dataset_lorenz import LorenzDataset
from utils.trainer import Trainer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    embed = 32
    lr = 1e-3
    epochs = 300

    sys.argv = sys.argv + ["--training_h5_file",
                           "./data/lorenz_training_rk.hdf5"]
    sys.argv = sys.argv + ["--eval_h5_file", "./data/lorenz_valid_rk.hdf5"]
    sys.argv = sys.argv + ["--train_batch_size", "16"]
    sys.argv = sys.argv + ["--stride", "64"]
    sys.argv = sys.argv + ["--n_train", "2048"]
    sys.argv = sys.argv + ["--save_steps", "25"]
    sys.argv = sys.argv + ["--n_eval", "16"]

    # Load model configuration
    config = PhysConfig(
        n_ctx=64,
        n_embd=embed,
        n_layer=4,
        n_head=4,
        state_dims=[3],
        activation_function="gelu_new",
        initializer_range=0.05,
    )

    # Load embedding model
    embedding_model = LorenzEmbedding(
        config
    )
    embedding_model.load_model(
        file_or_path_directory='./tests/koopman_git_2/koop_model/embedding_lorenz200.pth')
    embedding_model = embedding_model.to(device)

    # Load visualization utility class
    viz = LorenzViz(plot_dir='./tests/koopman_git_2/plots/')

    # Init transformer model
    transformer = PhysformerGPT2(config, "Lorenz")
    model = PhysformerTrain(config, transformer)
    # if(training_args.epoch_start > 0):
    #     model.load_model(training_args.ckpt_dir,
    #                      epoch=training_args.epoch_start)
    # if(model_args.transformer_file_or_path):
    #     model.load_model(model_args.transformer_file_or_path)

    # Initialize training and validation datasets
    training_loader = LorenzDataset(
        embedder=embedding_model,
        file_path='./tests/koopman_git_2/lorenz_data_train.h5',
        block_size=config.n_ctx,
        ndata=2048,
        stride=64,
    )
    testing_loader = LorenzDataset(
        embedder=embedding_model,
        file_path='./tests/koopman_git_2/lorenz_data_test.h5',
        block_size=256,
        stride=1024,
        ndata=16, 
        eval=True,
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 14, 2, eta_min=1e-9)

    args = TrainingArguments()
    args.exp_dir = './tests/koopman_git_2/log'
    args.src_device = device
    args.epoch_start = 0
    args.seed = 2
    args.epochs = epochs
    args.save_steps = 25
    args.ckpt_dir = './tests/koopman_git_2/koop_model/'
    args.train_batch_size = 16
    args.plot_max = 500


    trainer = Trainer(
        model,
        args,
        (optimizer, scheduler),
        train_dataset=training_loader,
        eval_dataset=testing_loader,
        embedding_model=embedding_model,
        viz=viz)

    trainer.train()
