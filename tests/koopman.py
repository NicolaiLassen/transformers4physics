from math import sin, cos, pi, tan
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

## TODO
## Implement batch size
## Check with actual physics system (like trajectory)
## Conditionally set eye to cuda

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

    def forward(self, x):
        M = self.L @ self.L.t() + self.eps * torch.eye(self.n*2).cuda()
        M11 = M[:self.n, :self.n]
        M12 = M[self.n:, :self.n]
        M21 = M[:self.n, self.n:]
        M22 = M[self.n:, self.n:]
        A = 2*(M11+M22+self.R-self.R.t())
        A = A.inverse() @ M21
        return A @ x


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


class Koopman(nn.Module):
    def __init__(self, features=16, hidden_dim=24, embed_dim=32):
        super().__init__()
        self.encoder = Encoder(
            in_features=features,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
        )
        self.ALR = ALR(
            n=embed_dim,
        )
        self.decoder = Decoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_features=features,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x, steps):
        x = x
        for _ in range(steps):
            x = self.ALR(x)
        return x


if __name__ == '__main__':
    features = 2
    hidden = 20
    embed = 10
    model = Koopman(features=features, hidden_dim=hidden, embed_dim=embed)
    model.cuda()
    model.train()
    x = torch.tensor([
        [sin(x), cos(x)] for x in np.arange(0, 2.05*pi, 0.1*pi)
    ], dtype=torch.float).cuda()
    seq_len = x.shape[0]
    alpha = 1000
    eps = 10**(-8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for i in range(301):
        T = len(x)

        l = torch.tensor(0).cuda()
        jrec = torch.tensor(0).cuda()
        for t in range(T):
            l = l + (model.encode(x[t]) -
                     model(model.encode(x[0]), t)).norm(p=2) ** 2
            jrec = jrec + \
                (x[t] - model.decode(model.encode(x[t]))).norm(p=2) ** 2

        l = 1/T * l
        jrec = 1/T * jrec
        loss = l + alpha * jrec
        loss.backward()

        optimizer.step()

        optimizer.zero_grad()
        if(i % 50 == 0):
            print('epoch {} loss: {}'.format(i, loss))
            model.eval()
            koops = torch.zeros(seq_len, embed).cuda()
            testout = torch.zeros(seq_len, features).cuda()
            koops[0] = model.encode(x[0])
            testout[0] = model.decode(model.encode(x[0]))
            for j in range(1, seq_len):
                koops[j] = model(koops[j-1], 1)
                testout[j] = model.decode(koops[j])
            error = F.mse_loss(testout, x)
            print('epoch {} error: {}'.format(i, error))
            model.train()
    test = x[0]
    test_len = seq_len
    koops = torch.zeros(test_len, embed).cuda()
    testout = torch.zeros(test_len, features).cuda()
    print(test)
    koops[0] = model.encode(x[0])
    testout[0] = model.decode(model.encode(x[0]))
    for i in range(1, test_len):
        koops[i] = model(koops[i-1], 1)
        testout[i] = model.decode(koops[i])
    print(testout)
