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

class SuperScaleFluidEmbedding(EmbeddingModel):
    """Embedding Koopman model for Landau-Lifshitz-Gilbert
    Args:
        config (EmbeddingConfig): Configuration class
    """

    model_name = "embedding_landau-lifshitz-gilbert"

    def __init__(self, config: EmmbedingConfig = None) -> None:
        super().__init__(config)

        # Learned Koopman operator
        self.diag_indices = np.diag_indices(config.embedding_dim)
        self.k_matrix_diag = nn.Sequential(
            nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, config.embedding_dim)
        )

        self.triu_indices = torch.triu_indices(config.embedding_dim, config.embedding_dim)
        self.tril_indices = torch.tril_indices(config.embedding_dim, config.embedding_dim)
        self.k_matrix_ut = nn.Sequential(
            nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, self.triu_indices[0].size(0))
        )
        self.k_matrix_lt = nn.Sequential(
            nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, self.tril_indices[0].size(0))
        )

        # Normalization occurs inside the model
        self.register_buffer("mu", torch.zeros(config.channels))
        self.register_buffer("std", torch.ones(config.channels))

    def forward(self, x: Tensor, field: Tensor) -> TensorTuple:
        """Forward pass
        Args:
            x (Tensor): [B, 3, H, W] Input feature tensor
            field (Tensor): [B, 2] the current external field
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

    def embed(self, x: Tensor, field: Tensor) -> Tensor:
        """Embeds tensor of state variables to Koopman observables
        Args:
            x (Tensor): [B, 3, H, W] Input feature tensor
            field (Tensor): [B, 3] the current external field
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

    def koopman_operation(self, g: Tensor, field: Tensor) -> Tensor:
        """Applies the learned Koopman operator on the given observables
        Args:
            g (Tensor): [B, config.n_embd] Koopman observables
        Returns:
            Tensor: [B, config.n_embd] Koopman observables at the next time-step
        """
        # Koopman operator
        k_matrix = Variable(
            torch.zeros(g.size(0), self.config.embedding_dim, self.config.embedding_dim)
        ).to(self.devices[0])
        # Populate the off diagonal terms
        k_matrix[:, self.triu_indices[0], self.triu_indices[1]] = self.k_matrix_ut(field)
        k_matrix[:, self.tril_indices[0], self.tril_indices[1]] = self.k_matrix_lt(field)

        # Populate the diagonal
        k_matrix[:, self.diag_indices[0], self.diag_indices[1]] = self.k_matrix_diag(field)

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
        x = (x - self.mu.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / self.std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return x

    def _unnormalize(self, x: Tensor) -> Tensor:
        return self.std[:3].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * x + self.mu[:3].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)


class SuperScaleFluidEmbeddingTrainer(EmbeddingTrainingHead):
    """Training head for the Lorenz embedding model
    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
        l1: Penalty weight for the reconstruction
        l2: Penalty weight for the dynamics in the koopman sequences
        l3: Penalty weight for the decay of the koopman operator, prevents overfitting
    """

    def __init__(self, embedding_model: EmbeddingModel, l1=1, l2=1e3, l3=1e-2):
        super().__init__()
        self.embedding_model = embedding_model
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def forward(self, states: Tensor, field: Tensor) -> FloatTuple:
        """Trains model for a single epoch
        Args:
            states (Tensor): [B, T, H, W] Time-series feature tensor
            field (Tensor): [B, 3] External fields (same across a batch)
        Returns:
            FloatTuple: Tuple containing:

                | (float): Koopman based loss of current epoch
                | (float): Reconstruction loss
        """
        self.embedding_model.train()
        return self._forward(states, field)

    def evaluate(self, states: Tensor, field: Tensor) -> FloatTuple:
        """Evaluates the embedding models reconstruction error and returns its
        predictions.
        Args:
            states (Tensor): [B, T, 3, H, W] Time-series feature tensor
            field (Tensor): [B, 3] External fields (same across a batch)
        Returns:
            FloatTuple: Tuple containing:

                | (float): Koopman based loss of current epoch
                | (float): Reconstruction loss
        """
        self.embedding_model.eval()
        return self._forward(states, field)

    def _forward(self, states: Tensor, field: Tensor):
        device = self.embedding_model.devices[0]

        loss_reconstruct = 0
        mseLoss = nn.MSELoss()

        xin0 = states[:, 0].to(device)  # Time-step

        # Model forward for initial time-step
        g0, xRec0 = self.embedding_model(xin0, field)

        loss = self.l2 * mseLoss(xin0, xRec0)
        loss_reconstruct = loss_reconstruct + self.l1 * mseLoss(xin0, xRec0).detach()

        g1_old = g0

        # Loop through time-series
        for t0 in range(1, states.shape[1]):
            xin0 = states[:, t0, :, :].to(device)  # Next time-step
            _, xRec1 = self.embedding_model(xin0, field)

            g1Pred = self.embedding_model.koopman_operation(g1_old, field)
            xgRec1 = self.embedding_model.recover(g1Pred)

            loss = (
                loss
                + self.l1 * mseLoss(xgRec1, xin0)
                + self.l2 * mseLoss(xRec1, xin0)
                + self.l3
                * torch.sum(torch.pow(self.embedding_model.koopman_operator, 2))
            )


            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach()
            g1_old = g1Pred

        return loss/states.size(1), loss_reconstruct/states.size(1)