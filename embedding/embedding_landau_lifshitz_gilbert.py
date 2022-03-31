from re import S
from turtle import st
from typing import Dict, List, Tuple

import numpy as np
import torch
from config.config_emmbeding import EmmbedingConfig
from torch import Tensor, nn
from torch.autograd import Variable

from embedding.embedding_model import EmbeddingModel, EmbeddingTrainingHead

from .backbone.conv_backbone import ConvBackbone
from .backbone.embedding_backbone import EmbeddingBackbone
from .backbone.restnet_backbone import ResnetBackbone
from .backbone.twins_svt_backbone import TwinsSVTBackbone

# Custom types
Tensor = torch.Tensor
TensorTuple = Tuple[torch.Tensor]
FloatTuple = Tuple[float]

backbone_models: Dict[str, EmbeddingBackbone] = {
    "Conv": ConvBackbone,
    "ResNet": ResnetBackbone,
    "TwinsSVT": TwinsSVTBackbone,
    # "Swin": SwinBackbone,
    # "vit": ViTBackbone
}

class LandauLifshitzGilbertEmbedding(EmbeddingModel):
    """Embedding Koopman model for Landau-Lifshitz-Gilbert
    Args:
        config (EmbeddingConfig): Configuration class
    """

    model_name = "embedding_landau-lifshitz-gilbert"

    def __init__(self, config: EmmbedingConfig = None) -> None:
        super().__init__(config)

        if config.backbone not in backbone_models.keys():
            raise NotImplementedError(
                "The {} backbone is not suppported".format(config.backbone))

        self.backbone: EmbeddingBackbone = backbone_models[config.backbone](
            img_size=config.image_size,
            backbone_dim=config.backbone_dim,
            embedding_dim=config.embedding_dim,
            fc_dim=config.fc_dim
        )
            
        # Learned Koopman operator
        self.k_matrix_diag = nn.Parameter(torch.ones(config.embedding_dim))

        # Off-diagonal indices
        xidx = []
        yidx = []
        for i in range(1, 10):
            yidx.append(np.arange(i, self.config.embedding_dim))
            xidx.append(np.arange(0, self.config.embedding_dim - i))

        self.xidx = torch.LongTensor(np.concatenate(xidx))
        self.yidx = torch.LongTensor(np.concatenate(yidx))
        self.k_matrix_ut = nn.Parameter(0.01 * torch.rand(self.xidx.size(0)))
        
        # Normalization occurs inside the model
        self.register_buffer('mu', torch.zeros(3))
        self.register_buffer('std', torch.ones(3))

    def forward(self, x: Tensor) -> TensorTuple:
        """Forward pass
        Args:
            x (Tensor): [B, 3, H, W] Input feature tensor
            field (Tensor): [B] the current external field
        Returns:
            (TensorTuple): Tuple containing:
                | (Tensor): [B, config.n_embd] Koopman observables
                | (Tensor): [B, 3, H, W] Recovered feature tensor
        """
        x = self._normalize(x)
        g = self.backbone.embed(x)
        # Decode
        out = self.backbone.recover(g)
        xhat = self._unnormalize(out)
        return g, xhat

    def embed(self, x: Tensor) -> Tensor:
        """Embeds tensor of state variables to Koopman observables
        Args:
            x (Tensor): [B, 3, H, W] Input feature tensor
        Returns:
            (Tensor): [B, config.n_embd] Koopman observables
        """
        x = self._normalize(x)
        g = self.backbone.embed(x)
        return g

    def recover(self, g: Tensor) -> Tensor:
        """Recovers feature tensor from Koopman observables
        Args:
            g (Tensor): [B, config.n_embd] Koopman observables
        Returns:
            (Tensor): [B, 3, H, W] Physical feature tensor
        """
        out = self.backbone.recover(g)
        x = self._unnormalize(out)
        return x

    def koopman_operation(self, g: Tensor) -> Tensor:
        """Applies the learned Koopman operator on the given observables
        Args:
            g (Tensor): [B, config.n_embd] Koopman observables
        Returns:
            Tensor: [B, config.n_embd] Koopman observables at the next time-step
        """
        # Koopman operator
        k_matrix = Variable(torch.zeros(
            g.size(0), self.config.embedding_dim, self.config.embedding_dim)).to(self.devices[0])
        # Populate the off diagonal terms
        k_matrix[:, self.xidx, self.yidx] = self.k_matrix_ut
        k_matrix[:, self.yidx, self.xidx] = -self.k_matrix_ut

        # Populate the diagonal
        ind = np.diag_indices(k_matrix.shape[1])
        k_matrix[:, ind[0], ind[1]] = self.k_matrix_diag

        # Apply Koopman operation
        g_next = torch.bmm(k_matrix, g.unsqueeze(-1))
        self.k_matrix = k_matrix

        return g_next.squeeze(-1)  # Squeeze empty dim from bmm

    @property
    def koopman_operator(self, requires_grad: bool = True) -> Tensor:
        """Current Koopman operator
        Args:
            requires_grad (bool, optional): If to return with gradient storage. Defaults to True
        Returns:
            Tensor: Full Koopman operator tensor
        """
        if not requires_grad:
            return self.k_matrix.detach()
        else:
            return self.k_matrix

    @property
    def koopman_diag(self):
        return self.k_matrix_diag

    def _normalize(self, x):
        x = (x - self.mu.view(1,-1,1,1)) / \
            self.std.view(1,-1,1,1)
        return x

    def _unnormalize(self, x: Tensor) -> Tensor:
        return self.std[:3].view(1,-1,1,1)*x +\
            self.mu[:3].view(1,-1,1,1)


class LandauLifshitzGilbertEmbeddingTrainer(EmbeddingTrainingHead):
    """Training head for the Lorenz embedding model
    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
    """

    def __init__(self, embedding_model: EmbeddingModel):
        super().__init__()
        self.embedding_model = embedding_model

    def forward(self, states: Tensor) -> FloatTuple:
        """Trains model for a single epoch
        Args:
            states (Tensor): [B, T, H, W] Time-series feature tensor
        Returns:
            FloatTuple: Tuple containing:

                | (float): Koopman based loss of current epoch
                | (float): Reconstruction loss
        """
        self.embedding_model.train()
        return self._forward(states) 

    def evaluate(self, states: Tensor) -> FloatTuple:
        """Evaluates the embedding models reconstruction error and returns its
        predictions.
        Args:
            states (Tensor): [B, T, 3, H, W] Time-series feature tensor
        Returns:
            FloatTuple: Tuple containing:

                | (float): Koopman based loss of current epoch
                | (float): Reconstruction loss
        """
        self.embedding_model.eval()
        return self._forward(states)

    def _forward(self, states: Tensor):
        device = self.embedding_model.devices[0]

        loss_reconstruct = 0
        mseLoss = nn.MSELoss()

        xin0 = states[:, 0].to(device)  # Time-step
    
        # Model forward for initial time-step
        g0, xRec0 = self.embedding_model(xin0)

        loss = (1e4)*mseLoss(xin0, xRec0)
        loss_reconstruct = loss_reconstruct + mseLoss(xin0, xRec0).detach()

        g1_old = g0

        # Loop through time-series
        for t0 in range(1, states.shape[1]):
            xin0 = states[:, t0, :, :].to(device)  # Next time-step
            _, xRec1 = self.embedding_model(xin0)

            g1Pred = self.embedding_model.koopman_operation(g1_old)
            xgRec1 = self.embedding_model.recover(g1Pred)

            loss = loss + mseLoss(xgRec1, xin0) + (1e4)*mseLoss(xRec1, xin0) \
                + (1e-1)*torch.sum(torch.pow(self.embedding_model.koopman_operator, 2))

            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach()
            g1_old = g1Pred
            
        return loss, loss_reconstruct