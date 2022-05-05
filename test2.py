import math
import os
from http.client import PRECONDITION_FAILED
from pickletools import optimize
from turtle import forward

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from genericpath import exists
from torch import einsum, nn

def exists(val):
    return val is not None

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :j]

        bias = torch.arange(j, device = device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))

        self.register_buffer('bias', bias, persistent = False)
        return qk_dots + self.bias

def extract_patches(tensor, tile_size=32, channels=3):
    kc, kh, kw = channels, tile_size, tile_size  # kernel size
    dc, dh, dw = channels, tile_size, tile_size  # stride
    patches = tensor.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    unfold_shape = patches.size()
    patches = patches.contiguous().view(patches.size(0), -1, kc, kh, kw)
    return patches, unfold_shape


def merge_patches(patches, unfold_shape):
    patches_orig = patches.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    patches_orig = patches_orig.view(patches.size(0), output_c, output_h, output_w)
    return patches_orig

# relative positional bias


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, alibi_num_heads=4, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.rel_pos =  nn.Identity # AlibiPositionalBias(alibi_num_heads)

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# classes

class SpatialFormer(nn.Module):
    """
    
    """
    
    def __init__(
        self,
        *,
        channels = 3,
        max_seq_len = 1024,
        patch_size = 32,
        alibi_num_heads = 2,
        heads=4,
        dim=512,
        dim_head=256,
        mlp_dim=512,
        dropout=0.
    ):
        super().__init__()


        self.layers = nn.ModuleList([])

        patch_dim = channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, alibi_num_heads=alibi_num_heads, heads = heads, dim_head = dim_head, dropout = dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        ]))

    def forward(self, x):
        # x: b, c, w, h
        x = self.to_patch_embedding(x)
        print(x.shape)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

import torch
from x_transformers import (ContinuousAutoregressiveWrapper,
                            ContinuousTransformerWrapper, Decoder)

if __name__ == '__main__':

    # to patches
    x = torch.arange(8)
    y = torch.arange(8)
    img, _ = torch.meshgrid(x, y)
    img = torch.stack([img.unsqueeze(0), img.unsqueeze(0)], dim=0)

    print(img.shape)
    pl, mapf = extract_patches(img.float(), 4, 1)
    print(mapf)
    print(pl.shape)
    print(merge_patches(pl, mapf).shape)

    t = ContinuousTransformerWrapper(
        dim_in= 224,
        dim_out= 224,
        max_seq_len=1024,
        attn_layers = Decoder(
            dim = 224,
            depth = 1,
            heads = 8,
            alibi_pos_bias = True,
            alibi_num_heads = 8
        )
    )
    
    # koopman
    m = nn.Linear(16, 224)
    s = [] 
    for i in range(pl.size(1)):
        patch = pl[:, i]
        patch = patch.view(-1, 16)
        latent = m(patch)
        s.append(latent)

    t(torch.stack(s, 1))



