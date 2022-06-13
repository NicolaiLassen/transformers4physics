from re import S
from turtle import st
from typing import Dict, List, Tuple
from einops import rearrange

import numpy as np
import torch
from config.config_emmbeding import EmmbedingConfig
from torch import Tensor, nn
from torch.autograd import Variable
import torch.nn.functional as F

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
                "The {} backbone is not suppported".format(config.backbone)
            )

        self.backbone: EmbeddingBackbone = backbone_models[config.backbone](
            img_size=[config.image_size_x, config.image_size_y],
            backbone_dim=config.backbone_dim,
            embedding_dim=config.embedding_dim,
            fc_dim=config.fc_dim,
            channels=config.channels,
        )

        self.use_koop_net = config.use_koop_net
        if self.use_koop_net:
            # Learned Koopman operator with net
            self.diag_indices = np.diag_indices(config.embedding_dim)
            self.k_matrix_diag = nn.Sequential(
                nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, config.embedding_dim)
            )

            # Off-diagonal indices
            if config.koopman_bandwidth < 0 or config.koopman_bandwidth >= config.embedding_dim:
                xidx = []
                yidx = []
                for i in range(1, config.embedding_dim):
                    xidx.append(np.arange(i, config.embedding_dim))
                    yidx.append(np.arange(0, config.embedding_dim - i))
                self.triu_indices = torch.LongTensor(
                    [np.concatenate(xidx), np.concatenate(yidx)]
                )
                self.tril_indices = torch.LongTensor(
                    [np.concatenate(yidx), np.concatenate(xidx)]
                )
            else:
                xidx = []
                yidx = []
                for i in range(1, config.koopman_bandwidth + 1):
                    xidx.append(np.arange(i, config.embedding_dim))
                    yidx.append(np.arange(0, config.embedding_dim - i))
                self.triu_indices = torch.LongTensor(
                    np.array([np.concatenate(xidx), np.concatenate(yidx)])
                )
                self.tril_indices = torch.LongTensor(
                    np.array([np.concatenate(yidx), np.concatenate(xidx)])
                )
            self.k_matrix_ut = nn.Sequential(
                nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, self.triu_indices[0].size(0))
            )
            self.k_matrix_lt = nn.Sequential(
                nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, self.tril_indices[0].size(0))
            )

        else:
            # Learned Koopman operator
            self.k_matrix_diag = nn.Parameter(torch.linspace(1,0,config.embedding_dim))
            self.diag_indices = np.diag_indices(config.embedding_dim)

            # Off-diagonal indices
            if config.koopman_bandwidth < 0 or config.koopman_bandwidth >= config.embedding_dim:
                xidx = []
                yidx = []
                for i in range(1, config.embedding_dim):
                    xidx.append(np.arange(i, config.embedding_dim))
                    yidx.append(np.arange(0, config.embedding_dim - i))
                self.triu_indices = torch.LongTensor(
                    [np.concatenate(xidx), np.concatenate(yidx)]
                )
                self.tril_indices = torch.LongTensor(
                    [np.concatenate(yidx), np.concatenate(xidx)]
                )
            else:
                xidx = []
                yidx = []
                for i in range(1, config.koopman_bandwidth + 1):
                    xidx.append(np.arange(i, config.embedding_dim))
                    yidx.append(np.arange(0, config.embedding_dim - i))
                self.triu_indices = torch.LongTensor(
                    np.array([np.concatenate(xidx), np.concatenate(yidx)])
                )
                self.tril_indices = torch.LongTensor(
                    np.array([np.concatenate(yidx), np.concatenate(xidx)])
                )
            self.k_matrix_ut = nn.Parameter(0.1*torch.rand(self.triu_indices[0].size(0)))
            self.k_matrix_lt = nn.Parameter(0.1*torch.rand(self.tril_indices[0].size(0)))

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
        x = torch.cat(
            [
                x,
                field[:,:1].unsqueeze(-1).unsqueeze(-1) * torch.ones_like(x[:, :1]),
                field[:,1:2].unsqueeze(-1).unsqueeze(-1) * torch.ones_like(x[:, :1]),
                # field[:,2:3].unsqueeze(-1).unsqueeze(-1) * torch.ones_like(x[:, :1]),
            ],
            dim=1,
        )
        x = self._normalize(x)
        g = self.backbone.embed(x)
        # Decode
        out = self.backbone.recover(g)
        xhat = self._unnormalize(out)
        if(not self.training):
            normsX = torch.sqrt(torch.einsum('ij,ij->j',xhat.swapaxes(1,3).reshape(-1,3).T, xhat.swapaxes(1,3).reshape(-1,3).T))
            normsX = normsX.reshape(-1,16,64).swapaxes(1,2)
            xhat[:,0,:,:] = x[:,0,:,:]/normsX
            xhat[:,1,:,:] = x[:,1,:,:]/normsX
            xhat[:,2,:,:] = x[:,2,:,:]/normsX
        return g, xhat

    def embed(self, x: Tensor, field: Tensor) -> Tensor:
        """Embeds tensor of state variables to Koopman observables
        Args:
            x (Tensor): [B, 3, H, W] Input feature tensor
            field (Tensor): [B, 3] the current external field
        Returns:
            (Tensor): [B, config.n_embd] Koopman observables
        """
        x = torch.cat(
            [
                x,
                field[:,:1].unsqueeze(-1).unsqueeze(-1) * torch.ones_like(x[:, :1]),
                field[:,1:2].unsqueeze(-1).unsqueeze(-1) * torch.ones_like(x[:, :1]),
                # field[:,2:3].unsqueeze(-1).unsqueeze(-1) * torch.ones_like(x[:, :1]),
            ],
            dim=1,
        )
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
        if(not self.training):
            normsX = torch.sqrt(torch.einsum('ij,ij->j',x.swapaxes(1,3).reshape(-1,3).T, x.swapaxes(1,3).reshape(-1,3).T))
            normsX = normsX.reshape(-1,16,64).swapaxes(1,2)
            x[:,0,:,:] = x[:,0,:,:]/normsX
            x[:,1,:,:] = x[:,1,:,:]/normsX
            x[:,2,:,:] = x[:,2,:,:]/normsX
        return x

    def koopman_operation(self, g: Tensor, field: Tensor = None) -> Tensor:
        """Applies the learned Koopman operator on the given observables
        Args:
            g (Tensor): [B, config.n_embd] Koopman observables
            field (Tensor): [B, 2] If the model is set to use field input in the koopman operation
        Returns:
            Tensor: [B, config.n_embd] Koopman observables at the next time-step
        """
        if self.use_koop_net:
            return self._koopman_operation_net(g, field)
        else:
            return self._koopman_op(g)

    def _koopman_op(self, g: Tensor):
        # Koopman operator
        k_matrix = Variable(
            torch.zeros(g.size(0), self.config.embedding_dim, self.config.embedding_dim)
        ).to(self.devices[0])
        # Populate the off diagonal terms
        k_matrix[:, self.triu_indices[0], self.triu_indices[1]] = self.k_matrix_ut
        k_matrix[:, self.tril_indices[0], self.tril_indices[1]] = self.k_matrix_lt

        # Populate the diagonal
        k_matrix[:, self.diag_indices[0], self.diag_indices[1]] = self.k_matrix_diag

        # Apply Koopman operation
        g_next = torch.bmm(k_matrix, g.unsqueeze(-1))
        self.k_matrix = k_matrix

        return g_next.squeeze(-1)  # Squeeze empty dim from bmm
        
    def _koopman_operation_net(self, g: Tensor, field: Tensor) -> Tensor:
        """Applies the learned Koopman operator on the given observables
        Args:
            g (Tensor): [B, config.n_embd] Koopman observables
        Returns:
            Tensor: [B, config.n_embd] Koopman observables at the next time-step
        """
        field = self._normalize_features(field)
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
        x = self.std[:3].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * x + self.mu[:3].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return x

    def _normalize_features(self, field):
        field = (field - self.mu[3:5].unsqueeze(0)) / self.std[3:5].unsqueeze(0)
        return field


class LandauLifshitzGilbertEmbeddingTrainer(EmbeddingTrainingHead):
    """Training head for the Lorenz embedding model
    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
        l1: Penalty weight for the dynamics in the koopman sequences
        l2: Penalty weight for the reconstruction
        l3: Penalty weight for the decay of the koopman operator, prevents overfitting
        l4: Penalty weight for ensuring unit vectors in the output
    """

    def __init__(self, embedding_model: EmbeddingModel, l1=1, l2=1e3, l3=1e-2, l4=1):
        super().__init__()
        self.embedding_model = embedding_model
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4

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
        normsX = xRec0.swapaxes(1,3).reshape(-1,3)
        normsX = torch.sqrt(torch.einsum('ij,ij->j',normsX.T, normsX.T))
        ones = torch.ones((normsX.shape[0])).to(device)

        loss = self.l2 * mseLoss(xin0, xRec0) + self.l4 * mseLoss(normsX, ones)
        loss_reconstruct = loss_reconstruct + self.l2 * mseLoss(xin0, xRec0).detach()

        g1_old = g0

        # Loop through time-series
        for t0 in range(1, states.shape[1]):
            xin0 = states[:, t0].to(device)  # Next time-step
            _, xRec1 = self.embedding_model(xin0, field)

            g1Pred = self.embedding_model.koopman_operation(g1_old, field)
            xgRec1 = self.embedding_model.recover(g1Pred)

            normsX = xRec1.swapaxes(1,3).reshape(-1,3)
            normsX = torch.sqrt(torch.einsum('ij,ij->j',normsX.T, normsX.T))
            ones = torch.ones((normsX.shape[0])).to(device)

            loss = (
                loss
                + self.l1 * mseLoss(xgRec1, xin0)
                + self.l2 * mseLoss(xRec1, xin0)
                + self.l3
                * torch.sum(torch.pow(self.embedding_model.koopman_operator, 2))
                + self.l4 * mseLoss(normsX, ones)
            )


            loss_reconstruct = loss_reconstruct + mseLoss(xRec1, xin0).detach()
            g1_old = g1Pred

        return loss/states.size(1), loss_reconstruct/states.size(1)

class LandauLifshitzGilbertEmbeddingTrainerNoDynamics(EmbeddingTrainingHead):
    """Training head for the Lorenz embedding model
    Args:
        config (PhysConfig): Configuration class with transformer/embedding parameters
        l2: Penalty weight for the reconstruction
        l4: Penalty weight for ensuring unit vectors in the output
    """

    def __init__(self, embedding_model: EmbeddingModel, l2=1e3, l4=1):
        super().__init__()
        self.embedding_model = embedding_model
        self.l2 = l2
        self.l4 = l4

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
        b, t, c, w, h = states.shape

        loss = 0
        loss_reconstruct = 0
        mseLoss = nn.MSELoss()

        for i in range(t):
            x = states[:,i]
            g = self.embedding_model.embed(x,field)
            xrec = self.embedding_model.recover(g)
            normsX = xrec.swapaxes(1,3).reshape(-1,3)
            normsX = torch.sqrt(torch.einsum('ij,ij->j',normsX.T, normsX.T))
            ones = torch.ones((normsX.shape[0])).to(device)

            loss = loss + self.l2 * mseLoss(x,xrec) + self.l4 * mseLoss(normsX, ones)
            loss_reconstruct = loss_reconstruct + self.l2 * mseLoss(x,xrec)

        return loss/t, loss_reconstruct/t