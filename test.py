from pickletools import optimize
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vit_pytorch import MAE, crossformer
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize, Compose

from embedding.backbone.conv_backbone import ConvBackbone

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

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

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

def split_tensor(tensor, tile_size=32):
    mask = torch.ones_like(tensor)
    # use torch.nn.Unfold
    stride  = tile_size//2
    unfold  = nn.Unfold(kernel_size=(tile_size, tile_size), stride=stride)

    # Apply to mask and original image
    mask_p  = unfold(mask)
    patches = unfold(tensor)
	
    patches = patches.reshape(3, tile_size, tile_size, -1).permute(3, 0, 1, 2)
    if tensor.is_cuda:
        patches_base = torch.zeros(patches.size(), device=tensor.get_device())
    else: 
        patches_base = torch.zeros(patches.size())
	
    tiles = []
    for t in range(patches.size(0)):
         tiles.append(patches[[t], :, :, :])

    return tiles, mask_p, patches_base, (tensor.size(2), tensor.size(3))

def rebuild_tensor(tensor_list, mask_t, base_tensor, t_size, tile_size=32):
    stride  = tile_size//2  
    # base_tensor here is used as a container

    for t, tile in enumerate(tensor_list):
         print(tile.size())
         base_tensor[[t], :, :] = tile  
	 
    base_tensor = base_tensor.permute(1, 2, 3, 0).reshape(3*tile_size*tile_size, base_tensor.size(0)).unsqueeze(0)
    fold = nn.Fold(output_size=(t_size[0], t_size[1]), kernel_size=(tile_size, tile_size), stride=stride)
   
    # https://discuss.pytorch.org/t/seemlessly-blending-tensors-together/65235/2?u=bowenroom
    output_tensor = fold(base_tensor)/fold(mask_t)
    # output_tensor = fold(base_tensor)
    return output_tensor

if __name__ == '__main__':

    image = Image.open('C:\\Users\\nicol\\OneDrive\\Desktop\\master\\transformers4physics\\dog.jpg')
    image = image.resize((128, 128), resample=0)
    x = TF.to_tensor(image).unsqueeze(0)
    

    print(x[0].shape)
    plt.imshow( x[0].permute(1, 2, 0)  )
    plt.show()

    tiles, _, _, _ = split_tensor(x, 32)

    fine_grained = ConvBackbone(img_size=32)

    patch_b = torch.stack(tiles)
    
    # shuffel idx
    for i in range(patch_b.size(0)):
        tile = tiles[i]
        fine_grained_o = fine_grained.embed(tile)
    
    print(fine_grained_o.shape)

import vit_pytorch.vit