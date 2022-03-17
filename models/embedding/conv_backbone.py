from abc import abstractmethod

import torch.nn as nn

# from models.embedding.embedding_backbone import EmbeddingBackbone


# TODO
class EmbeddingBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        scale_size=64,
        depth=6
    ):
        super().__init__()

        # activation
        activation = nn.SiLU(inplace=True)

        # observable net
        observable_net_layers = []
        observable_net_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=channels, kernel_size=1,
                      out_channels=scale_size),
            nn.BatchNorm2d(scale_size),
            activation
        ))
        for _ in range(depth):
            observable_net_layers.append(nn.Sequential(
                nn.Conv2d(in_channels=scale_size,
                          kernel_size=1,
                          out_channels=scale_size
                          ),
                nn.BatchNorm2d(32),
                activation
            ))

        # Recovery net
        recovery_net_layers = []
        recovery_net_layers.append(nn.Sequential(
            nn.Conv2d(in_channels=channels, kernel_size=1,
                      out_channels=scale_size),
            nn.BatchNorm2d(scale_size),
            activation
        ))

        ## create models
        self.observable_net_layers = nn.Sequential(*observable_net_layers)
        self.recovery_net_layers = nn.Sequential(*recovery_net_layers)



    def observableNet(self, x):
        return self.observable_net_layers(x)

    def observableNetFC(self):
        return super().observableNetFC()

    def recoveryNet(self, x):
        return self.recovery_net_layers(x)

    def recoveryNetFC(self):
            return super().recoveryNetFC()


    def forward(self, x):
        self.observableNet(x)


if __name__ == '__main__':
    # wandb sweep sweep_embed.yaml
    # wandb sweep autoregressive.yaml
    model = ConvBackbone()
    print(model)
