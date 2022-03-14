import torch
import torch.nn as nn

import torch.nn.functional as F

class ConvMToH(nn.Module):
    def __init__(self, res=224, in_channels=4, hidden_channels=[16, 32, 64], kernel_size=[4, 4, 3]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=hidden_channels[0],
                kernel_size=kernel_size[0],
            ),
            F.silu,
            nn.Conv3d(
                in_channels=hidden_channels[0],
                out_channels=hidden_channels[1],
                kernel_size=kernel_size[1],
            ),
            F.silu,
            nn.Conv3d(
                in_channels=hidden_channels[1],
                out_channels=hidden_channels[2],
                kernel_size=kernel_size[2],
            ),
            F.silu,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=hidden_channels[2],
                out_channels=hidden_channels[1],
                kernel_size=kernel_size[2],
            ),
            F.silu,
            nn.ConvTranspose3d(
                in_channels=hidden_channels[1],
                out_channels=hidden_channels[0],
                kernel_size=kernel_size[1],
            ),
            F.silu,
            nn.ConvTranspose3d(
                in_channels=hidden_channels[0],
                out_channels=in_channels,
                kernel_size=kernel_size[0],
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
