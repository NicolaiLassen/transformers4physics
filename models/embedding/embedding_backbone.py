from abc import abstractmethod
import torch.nn as nn

class EmbeddingBackbone(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        # Save config in model
        self.config = config

    @abstractmethod
    def observable_net(self):
        pass

    @abstractmethod
    def observable_net_fc(self):
        pass
    
    @abstractmethod
    def recovery_net(self):
        pass

    @abstractmethod
    def recovery_net_fc(self):
        pass