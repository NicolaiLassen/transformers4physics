from abc import abstractmethod
from typing import List
import torch.nn as nn

class EmbeddingBackbone(nn.Module):
    def __init__(self,
        channels: int = 3,
        img_size: List[int] = [32, 32],
        backbone_dim: int = 128,
        embedding_dim: int = 128,
        fc_dim: int = 128,
    ):
        super().__init__()
        # Save config in model
        
    @abstractmethod
    def observable_net(self, **kwargs):
        pass

    @abstractmethod
    def observable_net_fc(self, **kwargs):
        pass
    
    @abstractmethod
    def recovery_net(self, **kwargs):
        pass

    @abstractmethod
    def recovery_net_fc(self, **kwargs):
        pass
    
    @abstractmethod
    def embed(self, **kwargs):
        pass

    @abstractmethod
    def recover(self, **kwargs):
        pass