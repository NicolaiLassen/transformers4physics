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
    """ Conv backbone
        simple parametized 
    """

    def __init__(
        self,
        channels=3,
        scale_size=900
    ):
        super().__init__()

        # activation
        activation = nn.SiLU(inplace=True)

        # find number to fit
        # log_b c = k
        # b_stride = math.exp(math.log(image_shape[0]) / depth)
        # b_stride_ceil = math.ceil(b_stride)
        # log_scale_depth = 1

        # Observable net

        observable_net_layers = []
        observable_net_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=channels, stride=100,
                      kernel_size=1, out_channels=100),
            nn.BatchNorm2d(100),
            activation
        ))

        # Parametized loop for depth n
        in_channels = 100

        for i in range(depth - 1):
            observable_net_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, kernel_size=1,
                          stride=2, out_channels=100),
            ))
            observable_net_layers.append(activation)
            in_channels = 100
        

        self.observable_net_fc_layers = nn.Sequential()

        #

        # Recovery net
        recovery_net_layers = []
        recovery_net_layers.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=scale_size,
                               kernel_size=1, out_channels=scale_size),
            nn.BatchNorm2d(scale_size),
            activation
        ))
        in_channels = scale_size

        # create models
        self.observable_net_layers = nn.Sequential(*observable_net_layers)

        self.recovery_net_layers = nn.Sequential(*recovery_net_layers)
        self.recovery_net_fc_layers = nn.Sequential(*recovery_net_layers)

    def observable_net(self, x):
        return self.observable_net_layers(x)

    def observable_net_fc(self, x):
        return self.observable_net_fc_layers(x)

    def recovery_net(self, x):
        return self.recovery_net_layers(x)

    def recovery_net(self, x):
        return self.recovery_net_fc_layers(x)

    def forward(self, x):
        out = self.observable_net(x)
        print(out.shape)
        out = self.flatten(out)
        print(out.shape)
        out = self.observable_net_fc(out)
        print(out.shape)


if __name__ == '__main__':
    # wandb sweep sweep_embed.yaml
    # wandb sweep autoregressive.yaml
    model = ConvBackbone()
    input_test = torch.rand(1, 3, 32, 32)
    model(input_test)
