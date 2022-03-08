import argparse
import sys
import logging
import torch
from config.config_phys import PhysConfig
from config.args import TrainingArguments
from magtense_micro_test.magtense_micro_test_embedding import MicroMagnetEmbedding
from tests.koopman_git_2.data.dataset_magnet import MicroMagnetismDataset
from tests.koopman_git_2.transformer.phys_transformer_gpt2 import PhysformerGPT2
from tests.koopman_git_2.transformer.phys_transformer_helpers import PhysformerTrain
from tests.koopman_git_2.utils.trainer import Trainer
from tests.koopman_git_2.viz.viz_magnet import MicroMagViz
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    embed = 128
    lr = 1e-3
    epochs = 300

    # Load model configuration
    config = PhysConfig(
        n_ctx=16,
        n_embd=embed,
        n_layer=6,
        n_head=4,
        state_dims=[3,36,36],
        activation_function="gelu_new",
        initializer_range=0.05,
    )

    # Load embedding model
    embedding_model = MicroMagnetEmbedding(
        config
    )
    embedding_model.load_model(
        file_or_path_directory='./checkpoints/magtense_micro_test/embedding_model300.pth')
    embedding_model = embedding_model.to(device)

    # Load visualization utility class
    viz = MicroMagViz(plot_dir='./plots/magtense_micro_test/')
    b = np.load('magtense_micro_test\cube36_3d_coord.npy')
    b = np.swapaxes(b, 0, 1)
    b = b.reshape(3, 36, 36)
    b = np.swapaxes(b,1,2)
    viz.setCoords(b)

    # Init transformer model
    transformer = PhysformerGPT2(config, "MicroMagnetism")
    model = PhysformerTrain(config, transformer)
    # if(training_args.epoch_start > 0):
    #     model.load_model(training_args.ckpt_dir,
    #                      epoch=training_args.epoch_start)
    # if(model_args.transformer_file_or_path):
    #     model.load_model(model_args.transformer_file_or_path)

    # Initialize training and validation datasets
    training_loader = MicroMagnetismDataset(
        embedder=embedding_model,
        file_path='./magtense_micro_test/cube36_3d.h5',
        block_size=config.n_ctx,
        ndata=-1,
        stride=6,
    )
    testing_loader = MicroMagnetismDataset(
        embedder=embedding_model,
        file_path='./magtense_micro_test/cube36_3d.h5',
        block_size=100,
        stride=100,
        ndata=-1, 
        eval=True,
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1e-10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 14, 2, eta_min=1e-9)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

    args = TrainingArguments()
    args.exp_dir = './log/magtense_micro_test/'
    args.src_device = device
    args.epoch_start = 0
    args.seed = 2
    args.epochs = epochs
    args.save_steps = 25
    args.ckpt_dir = './checkpoints/magtense_micro_test/'
    args.train_batch_size = 4
    args.plot_max = 500


    trainer = Trainer(
        model,
        args,
        (optimizer, scheduler),
        train_dataset=training_loader,
        eval_dataset=testing_loader,
        embedding_model=embedding_model,
        # viz=viz
    )

    trainer.train()
