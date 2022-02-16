'''
From https://github.com/fletchf/skel 
'''

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from koopman_model import KoopmanModel
from lorenz_data import create_lorenz_dataset, create_lorenz_sequence, plot_lorenz
from optimizer import SimErrorOptimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def perform_test(model : KoopmanModel):
    test_lorenz = create_lorenz_sequence(
        x=10.9582,
        y=-2.4449,
        z=35.7579,
        steps=128,
    )
    model.eval()
    asd = len(test_lorenz)
    plot_lorenz(test_lorenz)
    test_pred_trajectory = []
    test_recon_true_trajectory = []
    y = torch.zeros((asd, embed)).cuda()
    test_init = [
        10.9582,
        -2.4449,
        35.7579,
    ]

    A = model()
    y[0] = model.phi(torch.tensor(test_init,dtype=torch.float64).cuda())
    test_pred_trajectory.append(test_init)
    test_recon_true_trajectory.append(test_init)
    for i in range(1, asd):
        y[i] = A @ y[i-1]
        test_pred_trajectory.append(model.phi_inv(y[i]).cpu().detach().numpy())
        test_recon_true_trajectory.append(model.phi_inv(model.phi(
            torch.tensor(test_lorenz[i], dtype=torch.float64).cuda())).cpu().detach().numpy())
    plot_lorenz(test_pred_trajectory)
    plot_lorenz(test_recon_true_trajectory)


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    dt = 0.01
    disc = True
    set_size = 200
    num_demos = set_size
    embed = 32

    data = create_lorenz_dataset(
        seed=42,
        set_size=set_size,
        dt=dt,
        num_steps=[1024, 512, 2048],
    )
    pos, pos_next = [], []
    seq_len = np.zeros((set_size, ), dtype=int)
    for i in range(set_size):
        y0 = data[i][0][:, :-1]
        y1 = data[i][0][:, 1:]
        len_y = data[i][1]
        pos.append(y0)
        pos_next.append(y1)

        seq_len[i] = len(y0)

    demo_len = seq_len[:num_demos]
    pred_horizon = np.min(demo_len)

    X = np.hstack(pos[:num_demos])
    Y = np.hstack(pos_next[:num_demos])

    scale_pos = np.max(np.abs(X), axis=-1, keepdims=True)

    X = X / scale_pos
    Y = Y / scale_pos

    model = KoopmanModel(
        X.shape[0],
        embed,
        recon=True,
        hidden_dims=[500],
    )
    model.cuda()

    optimizer = SimErrorOptimizer(
        model,
        [X.T, Y.T],
        [num_demos, demo_len],
        tvec=None,
        disc=True,
        errtype='sim',
        lr=0.001,
        epochs=1000,
        batch_size=20,
        debug=True,
        alpha=1000,
    )

    losses = optimizer.train()

    perform_test(model)
