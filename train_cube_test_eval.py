'''
From https://github.com/fletchf/skel 
'''

from re import X
import matplotlib as mpl
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import h5py
from config.config_phys import PhysConfig

from magtense_micro_test.magtense_micro_test_embedding import MicroMagnetEmbedding
from tests.koopman_git_2.transformer.phys_transformer_gpt2 import PhysformerGPT2
from tests.koopman_git_2.viz.viz_magnet import MicroMagViz
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mpl.use('TkAgg')

if __name__ == '__main__':
    embed = 128
    create_statics = False
    init_embeds = 4
    cfg = PhysConfig(
        n_ctx=16,
        n_embd=embed,
        n_layer=6,
        n_head=4,
        state_dims=[3,36,36],
        activation_function="gelu_new",
        initializer_range=0.05,
    )
    model = MicroMagnetEmbedding(
        cfg
    )
    model.load_model(
        file_or_path_directory='./checkpoints/magtense_micro_test/embedding_model300.pth')
    config = PhysConfig(
        n_ctx=16,
        n_embd=embed,
        n_layer=6,
        n_head=4,
        state_dims=[3,36,36],
        activation_function="gelu_new",
        initializer_range=0.05,
    )
    transformer = PhysformerGPT2(config, "MicroMagnetism")
    transformer = transformer.cuda()
    transformer.load_model(
        './checkpoints/magtense_micro_test/transformer_MicroMagnetism300.pth')
        
    viz = MicroMagViz(plot_dir='./plots/magtense_micro_test/final/')
    b = np.load('magtense_micro_test\cube36_3d_coord.npy')
    b = b.swapaxes(0,1).reshape(3,36,36).swapaxes(1,2)
    viz.setCoords(b)
    f = h5py.File('./magtense_micro_test/cube36_3d.h5', 'r')
    data = f['dataset_1']
    with torch.no_grad():
        model.eval()
        model = model.cuda()
        asd = len(data)
        data = torch.tensor(data,dtype=torch.float).cuda()

        Z = model.embed(data)
        Z_trans_in = Z[0:init_embeds].unsqueeze(0)
        Z_trans = transformer.generate(
            Z_trans_in, max_length=asd)
        Z_trans = Z_trans[0].reshape(-1, model.obsdim)

        test_transform = model.recover(Z_trans)
        test_recon = model.recover(model.embed(data))
        
        viz.plotPrediction(test_transform.clone().detach(), data.clone().detach(), plot_dir='pred_init_embeds_{}.gif'.format(init_embeds))

        if create_statics:
            viz.plotPrediction(data.clone().detach(), data.clone().detach(), plot_dir='target.gif')
            viz.plotPrediction(test_recon.clone().detach(), data.clone().detach(), plot_dir='recon.gif'.format(init_embeds))