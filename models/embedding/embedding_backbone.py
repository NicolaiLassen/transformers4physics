from abc import abstractmethod
import torch.nn as nn

class EmbeddingBackbone(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        # Save config in model
        self.config = config

    @abstractmethod
    def observableNet(self):
        pass

    @abstractmethod
    def observableNetFC(self):
        pass
    
    @abstractmethod
    def recoveryNet(self):
        pass

    @abstractmethod
    def recoveryNetFC(self):
        pass