'''
From https://github.com/fletchf/skel 
'''

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from koopman_model import KoopmanModel
from lorenz_data import create_lorenz_dataset, create_lorenz_sequence, plot_lorenz, average_lorenz_sequences, denormalize_lorenz_seq, normalize_lorenz_seq
from optimizer import SimErrorOptimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def perform_test(model: KoopmanModel, init=None):
    with torch.no_grad():
        if(type(init) == None):
            init = np.array([
                0,
                0,
                25,
            ])
        if(normalize):
            init = denormalize_lorenz_seq(
                np.array([init]), myx, myy, myz, stdx, stdy, stdz)[0]
        test_lorenz = create_lorenz_sequence(
            x=init[0],
            y=init[1],
            z=init[2],
            steps=128,
        )

        plot_lorenz(test_lorenz, title="True")

        if(normalize):
            init = normalize_lorenz_seq(np.array([init]),myx, myy, myz, stdx, stdy, stdz)[0]
            test_lorenz = normalize_lorenz_seq(
                test_lorenz, myx, myy, myz, stdx, stdy, stdz)

        model.eval()
        asd = len(test_lorenz)

        test_recon_true_trajectory = []
        Z = torch.zeros((asd, model.dim_K)).cuda()

        Z[0] = model.phi(torch.tensor(init, dtype=torch.float64).cuda())
        test_recon_true_trajectory.append((model.phi_inv(Z[0])).cpu().detach().numpy())
        for i in range(1, asd):
            Z[i] = model.phi(torch.tensor(
                test_lorenz[i], dtype=torch.float64).cuda())
            test_recon_true_trajectory.append(
                (model.phi_inv(Z[i])).cpu().detach().numpy())

        tvec = torch.arange(0, int(asd), device=device).unsqueeze(-1)
        z0 = model.phi(torch.tensor((init.T), device=device, dtype=torch.float64))
        A = model()
        lam, V = torch.linalg.eig(A)
        Lamt = torch.pow(lam.repeat(tvec.shape[0], 1), tvec)
        At = V @ (Lamt.diag_embed() @ V.inverse())
        Z_sim = (At.real @ z0.T).squeeze()

        X_sim = model.phi_inv(Z_sim).cpu().detach().numpy()

        if(normalize):
            X_sim = denormalize_lorenz_seq(
                X_sim, myx, myy, myz, stdx, stdy, stdz)
            test_recon_true_trajectory = denormalize_lorenz_seq(
                test_recon_true_trajectory, myx, myy, myz, stdx, stdy, stdz)


        if(normalize):
            test_lorenz = denormalize_lorenz_seq(
                test_lorenz, myx, myy, myz, stdx, stdy, stdz)

        print(test_lorenz[0:3])
        print(test_recon_true_trajectory[0:3])
        print(X_sim[0:3])
        
        mse = ((Z_sim-Z)**2).mean(axis=1)
        print(mse)
        print(mse.median())
        print(mse.mean())
        plot_lorenz(test_recon_true_trajectory,
                    title="Reconstructed step by step")
        plot_lorenz(X_sim, title="Koopman + recon")


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    dt = 0.01
    disc = True
    set_size = 100
    num_demos = set_size
    embed = 32
    num_steps = [256, 512]
    normalize = True

    if(normalize):
        data, myx, myy, myz, stdx, stdy, stdz = create_lorenz_dataset(
            seed=42,
            set_size=set_size,
            dt=dt,
            num_steps=num_steps,
            normalize=normalize,
        )
    else:
        data = create_lorenz_dataset(
            seed=42,
            set_size=set_size,
            dt=dt,
            num_steps=num_steps,
            normalize=normalize,
        )

    pos, pos_next = [], []
    seq_len = np.zeros((set_size, ), dtype=int)
    for i in range(set_size):
        y0 = data[i][0][:, :-1]
        y1 = data[i][0][:, 1:]
        len_y = data[i][0].shape[-1]
        pos.append(y0)
        pos_next.append(y1)

        seq_len[i] = y0.shape[1]

    demo_len = seq_len[:num_demos]

    X = np.hstack(pos[:num_demos])
    Y = np.hstack(pos_next[:num_demos])

    scale_pos = np.max(np.abs(X), axis=-1, keepdims=True)

    X = X / scale_pos
    Y = Y / scale_pos

    model = KoopmanModel(
        X.shape[0],
        embed,
        recon=True,
        hidden_dims=[50, 50],
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
        epochs=500,
        batch_size=5000,
        debug=True,
        alpha=1000,
    )

    losses = optimizer.train()

    # perform_test(model, init=data[0][0][:, 0])
    if(normalize):
        perform_test(model, init=normalize_lorenz_seq(np.array([data[0][0][:,0]]),myx, myy, myz, stdx, stdy, stdz)[0])
    else:
        perform_test(model, init=np.array(data[0][0][:,0]))