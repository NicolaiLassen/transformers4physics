import math
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from genericpath import exists
from pandas import merge
from torch import einsum, nn, unsqueeze
from x_transformers import (ContinuousAutoregressiveWrapper,
                            ContinuousTransformerWrapper, Decoder, Encoder)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# scalable vit

# helpers

def extract_patches(tensor, tile_size=32, channels=3):
    kc, kh, kw = channels, tile_size, tile_size
    dc, dh, dw = channels, tile_size, tile_size
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

# spatialvit

class SpatialViT(nn.Module):
    def __init__(
        self,
        *,
        spatial_encoder,
        spatial_decoder,
        autoregressive,
        channels=1,
        patch_size=64,
        encoder_max_seq_len=128,
        autoregressive_max_seq_len=32,
    ):
        super().__init__()

        self.channels = channels
        self.image_to_patches = lambda x : extract_patches(x, patch_size,  channels)
        self.patches_to_image = lambda e, unfold_shape : merge_patches(e, unfold_shape)

        # Encoder 
        self.spatial_encoder = spatial_encoder
        # Decoder
        self.spatial_decoder = spatial_decoder
        # Autoregressive
        self.autoregressive = autoregressive

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, **kwargs):
        e, unfold_shape = self.embed(start_tokens)

        image = self.recover(e, unfold_shape)
        return image
    
    def embed(self, x, **kwargs):
        # x: b, c, w, h
        pe, unfold_shape = self.image_to_patches(x)
        # e: b, dim
        e = self.spatial_encoder(pe)
        return e, unfold_shape

    def recover(self, e, unfold_shape):
        # e: b, dim
        self.spatial_decoder(e)
        # x: b, c, w, h
        image = self.patches_to_image(e, unfold_shape)
        return image

    def forward(self, x, **kwargs):
        _, s, _, _, _ = *x.shape, 

        loss = 0
        for t in range(s):
            xt = x[:, t]
            e, unfold_shape = self.embed(xt)
            d = self.recover(e, unfold_shape)
            loss = loss + F.mse_loss(xt, d)
        
        print(loss)
        print(e.shape)
        for i in range(e.size(2)):
            ei = e[:, :-1, i]
            eo = e[:, 1:, i]

            mask = kwargs.get('mask', None)
            if mask is not None and mask.shape[1] == x.shape[1]:
                mask = mask[:, :-1]
                kwargs['mask'] = mask

            out = self.autoregressive(ei, **kwargs)
            loss += F.mse_loss(out, eo)
    
        return loss / e.size(2)


if __name__ == '__main__':

    import torchvision.transforms.functional as TF
    from PIL import Image

    # to patches
    f = h5py.File(
        "C:\\Users\\nicol\\OneDrive\\Desktop\\master\\smoke\\solved_fluid.h5", "r")

    xdata = f.get('1')
    xdata = torch.tensor(np.array(xdata))[1:].float().cuda()
    model = SpatialViT(
        channels=1,
        spatial_encoder = nn.Identity(),
        spatial_decoder = nn.Identity(),                  
        autoregressive = ContinuousTransformerWrapper(
            dim_in = 256,
            dim_out = 256,
            max_seq_len = 256,
            attn_layers = Decoder(
                dim = 256,
                depth = 4,
                heads = 4
            )
        )
    ).cuda()


    img = torch.randn(1, 20, 1, 128, 128).cuda()

    preds = model(img)
    print(preds.shape)

    optimizer_transformer = optim.Adam(model.parameters(), lr=0.0001)

    for i in range(100):
        optimizer_transformer.zero_grad()
        loss = model(xdata)
        print(loss.shape)
        loss.backward()
        optimizer_transformer.step()
        print(loss)