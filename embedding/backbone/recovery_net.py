from torch import nn

class RecoveryNet:

    def __init__(self, backbone_dims):
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                backbone_dims[2],
                backbone_dims[1],
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="zeros",
                output_padding=1,
            ),
            nn.BatchNorm2d(backbone_dims[1]),
            nn.LeakyReLU(0.02, inplace=True),
            
            nn.ConvTranspose2d(
                backbone_dims[1],
                backbone_dims[0],
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="zeros",
                output_padding=1,
            ),
            nn.BatchNorm2d(backbone_dims[0]),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(
                backbone_dims[0],
                3,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="zeros",
                output_padding=1,
            ),
        )

