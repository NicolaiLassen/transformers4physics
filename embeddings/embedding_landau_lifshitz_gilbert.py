from torch import Tensor, nn
import torch
import numpy as np
from typing import List, Tuple
from torch.autograd import Variable
from config.config_phys import PhysConfig
from embeddings.embedding_model import EmbeddingModel, EmbeddingTrainingHead

TensorTuple = Tuple[torch.Tensor]

## TODO convert to attention based medthod
# https://arxiv.org/pdf/2110.06509.pdf

# Note
# We wan't a stable model for non-lin systems 
# how to generate time of magnetization
# data -> embed -> seq 
# s_0 -> ebmed -> (s_1) feed self with  

# Custom types
Tensor = torch.Tensor
TensorTuple = Tuple[torch.Tensor]
FloatTuple = Tuple[float]

class LandauLifshitzGilbertEmbedding(EmbeddingModel):
    """ Stable Koopman Embedding model for Landau Lifshitz Gilbert system """

    def __init__(self, config: PhysConfig) -> None:  
        super().__init__(config)

        # TODO Landau Lifshitz Gilbert system is a fully connected system so use attention
        self.observableNet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 5), stride=2, padding=2, padding_mode='zeros'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),
            # 8, 32, 32, 32
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            # 16, 16, 16, 16
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=2, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            # 
            # 32, 8, 8, 8
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=1, padding=0, padding_mode='zeros'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),
        )

        self.observableNetFC = nn.Sequential(
            nn.Linear(64*8*8, 8*8*8),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(8*8*8, config.n_embd),
            nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
        )

        self.recoveryNetFC = nn.Sequential(
            nn.Linear(config.n_embd, 8*8*8),
            nn.LeakyReLU(1.0, inplace=True),
            nn.Linear(8*8*8, 64*8*8),
            nn.LeakyReLU(0.02, inplace=True),
        )

        self.recoveryNet = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), stride=1, padding=0, padding_mode='circular'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            # 
            # 32, 8, 8, 8
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),
            # 
            # 16, 16, 16, 16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),
            # 8, 32, 32, 32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, padding_mode='circular'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(64, 3, kernel_size=(1, 1), stride=1, padding=0, padding_mode='circular')
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
        return g, xhat

    def embed(self, x: Tensor) -> Tensor:
        """Embeds tensor of state variables to Koopman observables
        Args:
            x (Tensor): [B, 2, H, W, D] Input feature tensor
        Returns:
            Tensor: [B, config.n_embd] Koopman observables
        """
        x = self._normalize(x)
        g0 = self.observableNet(x)
        print(g0.shape)
        g = self.observableNetFC(g0.view(g0.size(0),-1))
        return g

    def recover(self, g: Tensor) -> Tensor:
        """Recovers feature tensor from Koopman observables
        Args:
            g (Tensor): [B, config.n_embd] Koopman observables
        Returns:
            (Tensor): [B, 2, H, W, D] Physical feature tensor
        """
        out = self.recoveryNetFC(g).view(-1, 64, 8, 8)
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
        x = (x - self.mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return x

    def _unnormalize(self, x):
        return self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * x + self.mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


class LandauLifshitzGilbertEmbeddingTrainer(EmbeddingTrainingHead):
    """Training head for the Lorenz embedding model
    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """
    def __init__(self, config: PhysConfig):
        """Constructor method
        """
        super().__init__()
        self.embedding_model = LandauLifshitzGilbertEmbedding(config)

    def forward(self, states: Tensor) -> FloatTuple:
        """Trains model for a single epoch
        Args:
            states (Tensor): [B, T, res, res] Time-series feature tensor
        Returns:
            FloatTuple: Tuple containing:
            
                | (float): Koopman based loss of current epoch
                | (float): Reconstruction loss
        """
        self.embedding_model.train()
        device = self.embedding_model.devices[0]

        loss_reconstruct = 0
        mseLoss = nn.MSELoss()

        xin0 = states[:,0].to(device) # Time-step

        # Model forward for initial time-step
        g0, xRec0 = self.embedding_model(xin0)

        loss = (1e4)*mseLoss(xin0, xRec0)
        loss_reconstruct = loss_reconstruct + mseLoss(xin0, xRec0).detach()

        g1_old = g0
        # Loop through time-series
        for t0 in range(1, states.shape[1]):
            xin0 = states[:,t0,:,:].to(device) # Next time-step
            _, xRec1 = self.embedding_model(xin0)

            g1Pred = self.embedding_model.koopmanOperation(g1_old)
            xgRec1 = self.embedding_model.recover(g1Pred)

            loss = loss + mseLoss(xgRec1, xin0) + (1e4)*mseLoss(xRec1, xin0) \
                + (1e-1)*torch.sum(torch.pow(self.embedding_model.koopmanOperator, 2))

            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach()
            g1_old = g1Pred

        return loss, loss_reconstruct

    def evaluate(self, states: Tensor) -> Tuple[float, Tensor, Tensor]:
        """Evaluates the embedding models reconstruction error and returns its
        predictions.
        Args:
            states (Tensor): [B, T, res, res] Time-series feature tensor
        Returns:
            Tuple[Float, Tensor, Tensor]: Test error, Predicted states, Target states
        """
        self.embedding_model.eval()
        device = self.embedding_model.devices[0]

        mseLoss = nn.MSELoss()

        # Pull out targets from prediction dataset
        yTarget = states[:,1:].to(device)
        xInput = states[:,:-1].to(device)
        yPred = torch.zeros(yTarget.size()).to(device)

        # Test accuracy of one time-step
        for i in range(xInput.size(1)):
            xInput0 = xInput[:,i].to(device)
            g0 = self.embedding_model.embed(xInput0)
            yPred0 = self.embedding_model.recover(g0)
            yPred[:,i] = yPred0.squeeze().detach()

        test_loss = mseLoss(yTarget, yPred)

        return test_loss, yPred, yTarget