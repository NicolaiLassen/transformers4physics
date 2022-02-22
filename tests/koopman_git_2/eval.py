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
from data.lorenz_data import create_lorenz_dataset, create_lorenz_sequence, plot_lorenz, average_lorenz_sequences, denormalize_lorenz_seq, normalize_lorenz_seq
from embedding.embedding_lorenz import LorenzEmbedding
from transformer import PhysformerGPT2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mpl.use('TkAgg')

if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    embed = 32
    init_embeds = 1
    cfg = PhysConfig(
        n_ctx=64,
        n_embd=embed,
        n_layer=4,
        n_head=4,
        state_dims=[3],
        activation_function="gelu_new",
        initializer_range=0.05,
    )
    model = LorenzEmbedding(
        cfg
    )
    model.load_model(
        file_or_path_directory='./tests/koopman_git_2/koop_model/embedding_lorenz200.pth')
    config = PhysConfig(
        n_ctx=64,
        n_embd=embed,
        n_layer=4,
        n_head=4,
        state_dims=[3],
        activation_function="gelu_new",
        initializer_range=0.05,
    )
    transformer = PhysformerGPT2(config, "Lorenz")
    transformer = transformer.cuda()
    transformer.load_model(
        './tests/koopman_git_2/koop_model/transformer_Lorenz300.pth')
    with torch.no_grad():
        x,y,z = -10,10,20
        test_lorenz = create_lorenz_sequence(
            x=x,
            y=y,
            z=z,
            steps=128,
        )

        plot_lorenz(test_lorenz, title="True")

        model.eval()
        model = model.cuda()
        asd = len(test_lorenz)

        test_recon_true_trajectory = []
        test_transform = []
        Z = torch.zeros((asd, model.obsdim)).cuda()

        Z[0] = model.embed(torch.tensor(
            [x, y, z], dtype=torch.float).cuda())
        test_recon_true_trajectory.append(
            (model.recover(Z[0])).cpu().detach().numpy())
        test_transform.append(
            (model.recover(Z[0])).cpu().detach().numpy())
        for i in range(1, asd):
            Z[i] = model.embed(
                torch.tensor(
                    test_lorenz[i], dtype=torch.float).cuda()
            )

            test_recon_true_trajectory.append(
                (model.recover(Z[i])).cpu().detach().numpy()
            )

        Z_trans_in = Z[0:init_embeds].unsqueeze(0)
        # Z_trans_in = Z[0].unsqueeze(0).unsqueeze(0)
        Z_trans = transformer.generate(
            Z_trans_in, max_length=asd)
        Z_trans = Z_trans[0].reshape(-1, model.obsdim)

        test_transform = [(model.recover(z)).cpu().detach().numpy() for z in Z_trans]

        test_recon_true_trajectory = np.array(test_recon_true_trajectory)
        test_transform = np.array(test_transform)
        # print(((test_lorenz - test_recon_true_trajectory)**2).mean())
        # print(((test_lorenz - test_transform)**2).mean())
        # print(((test_recon_true_trajectory - test_transform)**2).mean())

        plot_lorenz(test_recon_true_trajectory.reshape(-1, 3),
                    title="Reconstructed step by step")
        plot_lorenz(test_transform.reshape(-1, 3),
                    title="Reconstructed using koopman transformer ({} initial koopman {})".format(init_embeds, 'embedding' if init_embeds == 1 else 'embeddings'))