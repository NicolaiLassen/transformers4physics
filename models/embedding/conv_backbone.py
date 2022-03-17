import math
import re
from abc import abstractmethod
from re import M, X
from statistics import mode

import torch
import torch.nn as nn
from numpy import mat


# TODO
class EmbeddingBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Save config in model

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

# https://www.sciencedirect.com/science/article/pii/S0304885319307978
# https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d#e276
# https://en.wikipedia.org/wiki/Swish_function
class ConvBackbone(EmbeddingBackbone):
    def __init__(
        self,
        channels=3,
        img_dim=32,
        embedding_dim=128,
        fc_layer=512
    ):
        super().__init__()
        
        final_patch_size = int(img_dim / 4)
        self.final_patch_size = final_patch_size
        self.embedding_dim = embedding_dim

        # Observable net
        # TODO: can be a for loop
        observable_net_fc_layers = []
        observable_net_fc_layers.append(nn.Sequential(         
            nn.Sigmoid()
        ))

        self.observable_net_fc_layers = nn.Sequential(
            nn.Linear(embedding_dim*final_patch_size**2, fc_layer),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(fc_layer, embedding_dim),
        )

        self.observable_net_layers = nn.Sequential(*observable_net_fc_layers)

        # Recovery net 
        self.recovery_net_fc_layers = nn.Sequential(
            nn.Linear(embedding_dim, fc_layer),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(fc_layer, embedding_dim*final_patch_size**2),
        )
        
        # TODO: can be a for loop
        recovery_net_layers = []
        recovery_net_layers.append(nn.Sequential(
            nn.Sigmoid()
        ))

        self.recovery_net_layers = nn.Sequential(*recovery_net_layers)
    

    def observable_net(self, x):
        return self.observable_net_layers(x)

    def observable_net_fc(self, x):
        return self.observable_net_fc_layers(x)

    def recovery_net(self, x):
        return self.recovery_net_layers(x)

    def recovery_net_fc(self, x):
        return self.recovery_net_fc_layers(x)

    def embed(self, x):
        out = self.observable_net(x)
        out = out.view(-1, self.embedding_dim*self.final_patch_size**2)
        out = self.observable_net_fc(out)
        return out

    def recover(self, x):
        out = self.recovery_net_fc(x)
        out = out.view(-1, self.embedding_dim, self.final_patch_size, self.final_patch_size)
        out = self.recovery_net(out)
        return out

    def forward(self, x):
        out = self.embed(x)
        out = self.recover(out)
        return out


if __name__ == '__main__':
    # wandb sweep sweep_embed.yaml
    # wandb sweep autoregressive.yaml
    model = ConvBackbone()
    input_test = torch.rand(1, 3, 32, 32)
    model(input_test)
