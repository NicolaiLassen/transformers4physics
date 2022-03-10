'''
From https://github.com/fletchf/skel 
'''

import argparse
import ast
from re import X
import matplotlib as mpl
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import h5py
from config.config_phys import PhysConfig
from magtense_micro_test.embed_config import read_config

from magtense_micro_test.magtense_micro_test_embedding import MicroMagnetEmbedding
from tests.koopman_git_2.transformer.phys_transformer_gpt2 import PhysformerGPT2
from tests.koopman_git_2.viz.viz_magnet import MicroMagViz
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mpl.use('TkAgg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    config_path = vars(args)['config']
    assert config_path is not None, 'Please provide a config file'
    cfg = read_config(config_path)
    cfg_tsf = cfg['TRANSFORMER']
    cfg_eval = cfg['EVAL']
    
    device = torch.device(
        "cuda:0" if cfg['META']['device'] == 'cuda' else "cpu")

    init_embeds = cfg_eval.getint('init_embeds')
    config = PhysConfig(
        n_ctx=cfg_tsf.getint('n_ctx'),
        n_embd=cfg['META'].getint('n_embd'),
        n_layer=cfg_tsf.getint('n_layer'),
        n_head=cfg_tsf.getint('n_head'),
        state_dims=ast.literal_eval(cfg['TRANSFORMER']['state_dims']),
        activation_function=cfg_tsf['activation_function'],
        initializer_range=cfg_tsf.getfloat('initializer_range'),
    )
    model = MicroMagnetEmbedding(
        config
    )
    model.load_model(
        file_or_path_directory=cfg['TRANSFORMER']['embed_model_path'])
    transformer = PhysformerGPT2(
        config,
        cfg['META']['model_name'],
    )
    transformer = transformer.to(device)
    transformer.load_model(cfg_eval['transformer_model_path'])

    viz = MicroMagViz(plot_dir=cfg_tsf['plot_dir'])
    viz.setCoords(cfg_tsf['viz_coords_path'])
    
    f = h5py.File(cfg_eval['data'], 'r')
    data = f['dataset_1']
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        asd = len(data)
        data = torch.tensor(data, dtype=torch.float).cuda()

        Z = model.embed(data)
        Z_trans_in = Z[0:init_embeds].unsqueeze(0)
        Z_trans = transformer.generate(
            Z_trans_in, max_length=asd)
        Z_trans = Z_trans[0].reshape(-1, model.obsdim)

        test_transform = model.recover(Z_trans)
        test_recon = model.recover(model.embed(data))

        viz.plotPrediction(test_transform.clone().detach(), data.clone(
        ).detach(), plot_dir='pred_init_embeds_{}.gif'.format(init_embeds))
        viz.plotPrediction(data.clone().detach(),
                            data.clone().detach(), plot_dir='target.gif')
        viz.plotPrediction(test_recon.clone().detach(), data.clone(
        ).detach(), plot_dir='recon.gif'.format(init_embeds))
