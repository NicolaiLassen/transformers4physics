from math import sin, cos, pi, tan
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F
import numpy as np
from lorenz_data import create_lorenz_data_loader, create_lorenz_dataset, create_lorenz_sequence, plot_lorenz

# TODO
# Try freezing encoder decoder and using a transformer to predict sequences


class Encoder(nn.Module):
    def __init__(self, in_features=16, hidden_dim=24, embed_dim=32):
        super().__init__()
        self.ff1 = nn.Linear(
            in_features=in_features,
            out_features=hidden_dim,
        )
        self.ff2 = nn.Linear(
            in_features=hidden_dim,
            out_features=embed_dim,
        )
        self.act = nn.ReLU()
        self.lnorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.ff1(x)
        x = self.act(x)
        x = self.ff2(x)
        x = self.act(x)
        x = self.lnorm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim=32, hidden_dim=24, out_features=16):
        super().__init__()
        self.ff1 = nn.Linear(
            in_features=embed_dim,
            out_features=hidden_dim,
        )
        self.ff2 = nn.Linear(
            in_features=hidden_dim,
            out_features=out_features,
        )
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.ff1(x)
        x = self.act(x)
        x = self.ff2(x)
        x = self.act(x)
        return x


class ALR(nn.Module):
    def __init__(self, n=32, eps=10**(-8)):
        super().__init__()
        self.n = n
        self.L = nn.Parameter(torch.rand((n*2, n*2)))
        self.R = nn.Parameter(torch.rand((n, n)))
        self.eps = eps

    def forward(self, x, steps=1):
        if(steps == 0):
            return x
        x = x.reshape(self.n, -1)
        M = self.L @ self.L.t() + self.eps * torch.eye(self.n*2).to(self.L.get_device())
        M11 = M[:self.n, :self.n]
        M12 = M[self.n:, :self.n]
        M21 = M[:self.n, self.n:]
        M22 = M[self.n:, self.n:]
        A = 2*(M11+M22+self.R-self.R.t())
        # Use eigenvalue decomp to quickly calc many steps at once
        # This is covered in https://arxiv.org/pdf/2110.06509.pdf in eq 29
        # Issues with complex numbers
        A = torch.linalg.inv(A) @ M21
        L, V = torch.linalg.eig(A)
        L = L ** steps
        At = V @ torch.diag_embed(L) @ torch.linalg.inv(V)
        x = x.type(torch.complex64)
        return (At @ x).reshape(-1, self.n)


class Koopman(nn.Module):
    def __init__(self, features, encoder, decoder, embed_dim=32, eps=10**(-8), use_cuda=True):
        super().__init__()
        self.encoder = encoder
        self.ALR = ALR(
            n=embed_dim,
            eps=eps,
        )
        self.decoder = decoder
        # I = torch.eye(features)
        # O = torch.zeros((features, features-embed_dim))
        # # if(use_cuda):
        # #     I = I.cuda()
        # #     O = O.cuda()
        # print(O.shape)
        # self.C = (I@O - O@I).t()
        # print(self.C.shape)

    def encode(self, x) -> torch.Tensor:
        # b, l, _ = x.shape
        # x = x.reshape(3,-1)
        # cx = self.C@x
        # cx = cx.reshape(b, l, -1)
        # x = x.reshape(b, l, -1)
        # return cx + self.encoder(x)
        return self.encoder(x)

    def decode(self, x) -> torch.Tensor:
        return self.decoder(x)

    def forward(self, x, steps=1):
        x = self.ALR(x, steps=steps)
        return x


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask

# TODO
# Implement multiplication by 1/t in the summation


def koopman_loss(og_sequences, encoded_sequences, predicted_encoded, reconstructed_sequences, Ts, alpha):
    device = og_sequences.get_device()
    loss = torch.tensor(0).to(device)
    mask = length_to_mask(Ts)
    jse = ((encoded_sequences-predicted_encoded)[mask]).norm(p=2) ** 2
    jrec = ((og_sequences-reconstructed_sequences)[mask]).norm(p=2) ** 2
    loss = jse + alpha * jrec
    loss = loss * 1/len(Ts)
    return loss


def perform_test(model):
    test_lorenz = create_lorenz_sequence(
        x=10.9582,
        y=-2.4449,
        z=35.7579,
        steps=150,
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

    y[0] = model.encode(torch.tensor(test_init).cuda())
    test_pred_trajectory.append(test_init)
    test_recon_true_trajectory.append(test_init)
    for i in range(1, asd):
        y[i] = model(y[i-1])
        test_pred_trajectory.append(model.decode(y[i]).cpu().detach().numpy())
        test_recon_true_trajectory.append(model.decode(model.encode(
            torch.tensor(test_lorenz[i], dtype=torch.float).cuda())).cpu().detach().numpy())
    plot_lorenz(test_pred_trajectory)
    plot_lorenz(test_recon_true_trajectory)


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    features = 3
    hidden = 500
    embed = 32
    alpha = 1000
    eps = 10**(-8)

    model = Koopman(
        features=features,
        encoder=Encoder(
            in_features=features,
            hidden_dim=hidden,
            embed_dim=embed,
        ),
        decoder=Decoder(
            embed_dim=embed,
            hidden_dim=hidden,
            out_features=features,
        ),
        embed_dim=embed,
    )
    model.cuda()

    data_set = create_lorenz_dataset(
        seed=42,
        set_size=1,
        num_steps=[511, 63, 255],
        x=[-1,1],
        y=[-1,1],
        z=[-1,1],
    )
    data_loader = create_lorenz_data_loader(
        data_set,
        batch_size=1,
        use_cuda=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for e in range(21):
        for (x, l) in data_loader:
            T = x.shape[1]
            device = x.get_device()
            x_encoded = model.encode(x)
            x_recon = model.decode(x_encoded)
            x_encoded_pred = torch.zeros(x_encoded.shape).to(device)
            for i in range(T):
                x_encoded_pred[:, i, :] = model(x_encoded[:, 0, :], steps=i)

            loss = koopman_loss(
                og_sequences=x,
                encoded_sequences=x_encoded,
                predicted_encoded=x_encoded_pred,
                reconstructed_sequences=x_recon,
                Ts=l,
                alpha=alpha,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if(e % 1 == 0):
            print('epoch {} loss: {}'.format(e, loss))

    print(model.ALR.R)
    print(model.ALR.L)
    perform_test(model)
