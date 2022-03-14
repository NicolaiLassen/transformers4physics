import argparse
import ast
import logging
import sys

import numpy as np
import torch

from config.args import TrainingArguments
from config.config_phys import PhysConfig
from data_utils.dataset_magnet import MicroMagnetismDataset
from magtense_micro_test.embed_config import read_config
from magtense_micro_test.magtense_micro_test_embedding import \
    MicroMagnetEmbedding
from models.transformer.phys_transformer_gpt2 import PhysformerGPT2
from models.transformer.phys_transformer_helpers import PhysformerTrain
from tests.koopman_git_2.utils.trainer import Trainer


from viz.viz_magnet import MicroMagViz

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
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
    cfg_tsf = cfg['TRANSFORMER']

    device = torch.device(
        "cuda:0" if cfg['META']['device'] == 'cuda' else "cpu")
    # Load model configuration
    config = PhysConfig(
        n_ctx=cfg_tsf.getint('n_ctx'),
        n_embd=cfg['META'].getint('n_embd'),
        n_layer=cfg_tsf.getint('n_layer'),
        n_head=cfg_tsf.getint('n_head'),
        state_dims=ast.literal_eval(cfg['TRANSFORMER']['state_dims']),
        activation_function=cfg_tsf['activation_function'],
        initializer_range=cfg_tsf.getfloat('initializer_range'),
    )

    # Load embedding model
    embedding_model = MicroMagnetEmbedding(
        config
    )
    embedding_model.load_model(
        file_or_path_directory=cfg_tsf['embed_model_path'],
    )
    embedding_model = embedding_model.to(device)

    # Load visualization utility class
    viz = MicroMagViz(plot_dir=cfg_tsf['plot_dir'])
    viz.setCoords(cfg_tsf['viz_coords_path'])

    # Init transformer model
    transformer = PhysformerGPT2(config, cfg['META']['model_name'])
    model = PhysformerTrain(config, transformer)
    if(cfg_tsf.getint('epoch_start') > 0):
        model.load_model(
            cfg['META']['checkpoint_dir'],
            epoch=cfg_tsf.getint('epoch_start'),
        )

    # Initialize training and validation datasets
    training_loader = MicroMagnetismDataset(
        embedder=embedding_model,
        file_path=cfg['META']['h5_file_train'],
        block_size=cfg_tsf.getint('block_size_train'),
        ndata=cfg_tsf.getint('ndata_train'),
        overwrite_cache=cfg_tsf.getboolean('overwrite_cached_data'),
        stride=cfg_tsf.getint('stride_train'),
    )
    testing_loader = MicroMagnetismDataset(
        embedder=embedding_model,
        file_path=cfg['META']['h5_file_eval'],
        block_size=cfg_tsf.getint('block_size_eval'),
        stride=cfg_tsf.getint('stride_eval'),
        ndata=cfg_tsf.getint('ndata_eval'),
        eval=True,
        overwrite_cache=cfg_tsf.getboolean('overwrite_cached_data'),
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg_tsf.getfloat('lr'), weight_decay=1e-10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 14, 2, eta_min=1e-9)

    args = TrainingArguments()
    args.exp_dir = cfg['META']['exp_dir'] + '/transformer/'
    args.src_device = device
    args.epoch_start = cfg_tsf.getint('epoch_start')
    args.seed = cfg_tsf.getint('seed')
    args.epochs = cfg_tsf.getint('epochs')
    args.save_steps = cfg_tsf.getint('save_steps')
    args.ckpt_dir = cfg['META']['checkpoint_dir']
    args.train_batch_size = cfg_tsf.getint('batch_size_train')
    args.eval_batch_size = cfg_tsf.getint('batch_size_eval')
    args.plot_max = cfg_tsf.getint('plot_max')

    trainer = Trainer(
        model,
        args,
        (optimizer, scheduler),
        train_dataset=training_loader,
        eval_dataset=testing_loader,
        embedding_model=embedding_model,
        viz=viz if cfg_tsf.getboolean('use_viz') else None
    )

    trainer.train()
