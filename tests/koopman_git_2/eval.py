'''
From https://github.com/fletchf/skel 
'''

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import h5py

from config_phys import PhysConfig
from lorenz_data import create_lorenz_dataset, create_lorenz_sequence, plot_lorenz, average_lorenz_sequences, denormalize_lorenz_seq, normalize_lorenz_seq
from embedding_lorenz import LorenzEmbedding
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    cfg = PhysConfig(
        n_ctx=64,
        n_embd=32,
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
        file_or_path_directory='./tests/koopman_git_2/koop_model/embedding_lorenz225.pth')
    f = h5py.File('lorenz_norm_params.h5', 'r')
    mu = torch.tensor(f['dataset_1'][0]).cuda()
    std = torch.tensor(f['dataset_1'][1]).cuda()
    f.close()
    f = h5py.File('koopman_op.h5', 'r')
    A = torch.tensor(f['dataset_1']).cuda()
    f.close()
    print(A)
    model.mu = mu
    model.std = std
    with torch.no_grad():
        test_lorenz = create_lorenz_sequence(
            x=15,
            y=15,
            z=30,
            steps=128,
        )

        plot_lorenz(test_lorenz, title="True")

        model.eval()
        model = model.cuda()
        asd = len(test_lorenz)

        test_recon_true_trajectory = []
        Z = torch.zeros((asd, model.obsdim)).cuda()

        Z[0] = model.embed(torch.tensor([15,15,30], dtype=torch.float).cuda())
        test_recon_true_trajectory.append(
            (model.recover(Z[0])).cpu().detach().numpy())
        for i in range(1, asd):
            Z[i] = model.embed(torch.tensor(
                test_lorenz[i], dtype=torch.float).cuda())
            test_recon_true_trajectory.append(
                (model.recover(Z[i])).cpu().detach().numpy())
        test_recon_true_trajectory = np.array(test_recon_true_trajectory)
        
        Z_koop = torch.zeros((asd, model.obsdim)).cuda()
        X_sim = torch.zeros((asd, 3))

        Z_koop[0] = Z[0]
        X_sim[0] = model.recover(Z[0])
        for i in range(1, asd):
            Z_koop[i] = A @ Z_koop[i-1]
            X_sim[i] = model.recover(Z_koop[i]).cpu()

        plot_lorenz(test_recon_true_trajectory.reshape(-1,3),
                    title="Reconstructed step by step")
        plot_lorenz(X_sim.reshape(-1,3), title="Koopman + recon")
