import math
from abc import abstractmethod
from statistics import mode

import torch
import torch.nn as nn


# TODO
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
    
# https://www.sciencedirect.com/science/article/pii/S0304885319307978
# https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d#e276
# https://en.wikipedia.org/wiki/Swish_function
class ConvBackbone(EmbeddingBackbone):
    """ Conv backbone
        simple parametized 
    """

    def __init__(
        self,
        channels=3,
        image_shape=(32, 32),
        scale_size=512,
        depth=6
    ):
        super().__init__()

        # activation
        activation = nn.SiLU(inplace=True)

        # Observable net
        print(math.log(scale_size) * depth)
        scale_starte = int(scale_size / depth)
        observable_net_layers = []
        observable_net_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=channels, stride=1,
                      kernel_size=1, out_channels=scale_starte),
            nn.BatchNorm2d(scale_starte),
            activation
        ))

        # Parametized loop for depth n
        in_channels = scale_starte
        for _ in range(depth):
            out_channels = in_channels*2
            observable_net_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, kernel_size=2, stride=1, out_channels=out_channels),
                nn.BatchNorm2d(out_channels),
                activation
            ))
            in_channels = out_channels
            print(in_channels)

        # FC
        
        # Recovery net
        recovery_net_layers = []
        recovery_net_layers.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=scale_size, kernel_size=1, out_channels=scale_size),
            nn.BatchNorm2d(scale_size),
            activation
        ))
        in_channels = scale_size
        for _ in range(depth):
            out_channels = in_channels*2
            recovery_net_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, kernel_size=2, stride=1, out_channels=out_channels),
                nn.BatchNorm2d(out_channels),
                activation
            ))
            in_channels = out_channels
            print(in_channels)

        # create models
        self.observable_net_layers = nn.Sequential(*observable_net_layers)

        self.recovery_net_layers = nn.Sequential(*recovery_net_layers)

    def observable_net(self):
        return super().observable_net()

    def observable_net_fc(self):
        return super().observable_net_fc()

    def recovery_net(self):
        return super().recovery_net()

    def recovery_net(self):
        return super().recovery_net()

    def forward(self, x):
        out = self.observableNet(x)
        print(out.shape)


if __name__ == '__main__':
    # wandb sweep sweep_embed.yaml
    # wandb sweep autoregressive.yaml
    model = ConvBackbone()
    input_test = torch.rand(1, 3, 32, 32)
    model(input_test)
