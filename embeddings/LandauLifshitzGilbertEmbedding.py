from torch import Tensor, nn
import torch
import numpy as np
from typing import List, Tuple
from torch.autograd import Variable

TensorTuple = Tuple[torch.Tensor]

## TODO convert to attention based medthod
# https://arxiv.org/pdf/2110.06509.pdf

# Note
# We wan't a stable model for non-lin systems 
# how to generate time of magnetization
# data -> embed -> seq 
# s_0 -> ebmed -> (s_1) feed self with  

class LandauLifshitzGilbertEmbedding(nn.Module):
    """ Stable Koopman Embedding model for Landau Lifshitz Gilbert system """

    def __init__(self, config: Config) -> None:  
        super().__init__(config)

        # TODO Landau Lifshitz Gilbert system is a fully connected system so use attention
        self.observableNet = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=(5, 5, 5), stride=2, padding=2, padding_mode='circular'),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.02, inplace=True),
            # 8, 32, 32, 32
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=2, padding=1, padding_mode='circular'),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.02, inplace=True),
            # 16, 16, 16, 16
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=2, padding=1, padding_mode='circular'),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.02, inplace=True),
            # 
            # 32, 8, 8, 8
            nn.Conv3d(128, 64, kernel_size=(1, 1, 1), stride=1, padding=0, padding_mode='circular'),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.02, inplace=True),
        )

        self.observableNetFC = nn.Sequential(
            nn.Linear(64*4*4*4, 8*4*4*4),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(8*4*4*4, config.n_embd),
            nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
        )

        self.recoveryNetFC = nn.Sequential(
            nn.Linear(config.n_embd, 8*4*4*4),
            nn.LeakyReLU(1.0, inplace=True),
            nn.Linear(8*4*4*4, 64*4*4*4),
            nn.LeakyReLU(0.02, inplace=True),
        )

        self.recoveryNet = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(1, 1, 1), stride=1, padding=0, padding_mode='circular'),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.02, inplace=True),
            # 
            # 32, 8, 8, 8
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.02, inplace=True),
            # 
            # 16, 16, 16, 16
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.02, inplace=True),
            # 8, 32, 32, 32
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv3d(64, 2, kernel_size=(1, 1, 1), stride=1, padding=0, padding_mode='circular')
        )

        # Learned Koopman operator
        self.kMatrixDiag = nn.Parameter(torch.ones(config.n_embd))

        # Off-diagonal indices
        xidx = []
        yidx = []
        for i in range(1, 10):
            yidx.append(np.arange(i, self.config.n_embd))
            xidx.append(np.arange(0, self.config.n_embd - i))

        self.xidx = torch.LongTensor(np.concatenate(xidx))
        self.yidx = torch.LongTensor(np.concatenate(yidx))
        self.kMatrixUT = nn.Parameter(0.01 * torch.rand(self.xidx.size(0)))

        # Normalization occurs inside the model
        self.register_buffer('mu', torch.tensor(0.))
        self.register_buffer('std', torch.tensor(1.))


def forward(self, x: Tensor) -> TensorTuple:
    """Forward pass
    Args:
        x (Tensor): [B, 2, H, W, D] Input feature tensor
    Returns:
        TensorTuple: Tuple containing:
            | (Tensor): [B, config.n_embd] Koopman observables
            | (Tensor): [B, 2, H, W, D] Recovered feature tensor
    """
    # Encode
    x = self._normalize(x)
    g0 = self.observableNet(x)
    g = self.observableNetFC(g0.view(g0.size(0),-1))
    # Decode
    out0 = self.recoveryNetFC(g).view(-1, 64, 4, 4, 4)
    out = self.recoveryNet(out0)
    xhat = self._unnormalize(out)
    return g, xhat, g0, out0, self._unnormalize(self.recoveryNet(g0))

def embed(self, x: Tensor) -> Tensor:
    """Embeds tensor of state variables to Koopman observables
    Args:
        x (Tensor): [B, 2, H, W, D] Input feature tensor
    Returns:
        Tensor: [B, config.n_embd] Koopman observables
    """
    x = self._normalize(x)
    g0 = self.observableNet(x)
    g = self.observableNetFC(g0.view(g0.size(0),-1))
    return g

def recover(self, g: Tensor) -> Tensor:
    """Recovers feature tensor from Koopman observables
    Args:
        g (Tensor): [B, config.n_embd] Koopman observables
    Returns:
        (Tensor): [B, 2, H, W, D] Physical feature tensor
    """
    out = self.recoveryNetFC(g).view(-1, 64, 4, 4, 4)
    out = self.recoveryNet(out)
    x = self._unnormalize(out)
    return x

def koopmanOperation(self, g: Tensor) -> Tensor:
    """Applies the learned Koopman operator on the given observables
    Args:
        g (Tensor): [B, config.n_embd] Koopman observables
    Returns:
        (Tensor): [B, config.n_embd] Koopman observables at the next time-step
    """
    # Koopman operator
    kMatrix = Variable(torch.zeros(g.size(0), self.config.n_embd, self.config.n_embd)).to(self.devices[0])
    # Populate the off diagonal terms
    kMatrix[:, self.xidx, self.yidx] = self.kMatrixUT
    kMatrix[:, self.yidx, self.xidx] = -self.kMatrixUT
    # Populate the diagonal
    ind = np.diag_indices(kMatrix.shape[1])
    kMatrix[:, ind[0], ind[1]] = self.kMatrixDiag

    # Apply Koopman operation
    gnext = torch.bmm(kMatrix, g.unsqueeze(-1))
    self.kMatrix = kMatrix

    return gnext.squeeze(-1) # Squeeze empty dim from bmm

@property
def koopmanOperator(self, requires_grad: bool =True) -> Tensor:
    """Current Koopman operator
    Args:
        requires_grad (bool, optional): If to return with gradient storage. Defaults to True
    Returns:
        Tensor: Full Koopman operator tensor
    """
    if not requires_grad:
        return self.kMatrix.detach()
    else:
        return self.kMatrix

@property
def koopmanDiag(self):
    return self.kMatrixDiag

def _normalize(self, x):
    x = (x - self.mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) / self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return x

def _unnormalize(self, x):
    return self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x + self.mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
