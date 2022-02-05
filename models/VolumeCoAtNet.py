import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

class VolumeCoAtNet(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

    def _make_layer(self):
        return
