from abc import abstractmethod
import torch.nn as nn
import vit_pytorch.twins_svt

class EmbeddingBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__( **kwargs)
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