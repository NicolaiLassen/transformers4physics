import math
import os
from http.client import PRECONDITION_FAILED
from pickletools import optimize
from turtle import forward
from matplotlib import patches

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

import h5py
import torch
from x_transformers import (ContinuousAutoregressiveWrapper,
                            ContinuousTransformerWrapper, Decoder)
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class KoopSim(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=4,
                      padding=2, padding_mode="zeros"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2,
                      padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2,
                      padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2,
                      padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(128, 256, kernel_size=2, stride=2,
                      padding=0, padding_mode="zeros"),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02, inplace=True),
        )

        self.encoder_fc = nn.Sequential(
                nn.Linear(1024, 256),
                nn.LeakyReLU(0.02, inplace=True),
                nn.Linear(256, 256),
                nn.LayerNorm(256, eps=1e-5),
            )

        self.decoder_fc = nn.Sequential(
                nn.Linear(256, 256),
                nn.LeakyReLU(0.02, inplace=True),
                nn.Linear(256, 1024),
                nn.LayerNorm(1024, eps=1e-5),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2,
                padding=1, output_padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),

            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2,
                      padding=1, output_padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.02, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2,
                      padding=1, output_padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2,
                      padding=1, output_padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2,
                      padding=1, output_padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2,
                      padding=1, output_padding=1, padding_mode="zeros")
        )
        
    def forward(self, x):

        e = self.encoder(x)
        e = e.view(-1, 256*2*2)
        e = self.encoder_fc(e)

        d = self.decoder_fc(e)
        d = d.view(-1, 256, 2, 2)
        d = self.decoder(d)
        return e, d

if __name__ == '__main__':

    from PIL import Image
    import torchvision.transforms.functional as TF

    image = Image.open('C:\\Users\\nicol\\OneDrive\\Desktop\\master\\transformers4physics\\dog.jpg')
    x = TF.to_tensor(image)
    x.unsqueeze_(0)

        # to patches
    f = h5py.File("C:\\Users\\nicol\\OneDrive\\Desktop\\master\\smoke\\solved_fluid.h5", "r")

    xdata = f.get('1')
    xdata = torch.tensor(xdata)[1:]

    print(xdata.shape)
    print(x[0][1][:128, :128].unsqueeze(0).unsqueeze(0).shape)
    p, un_shape = extract_patches(x[0][1][:256 * 4, :256 * 4].unsqueeze(0).unsqueeze(0), 128, 1)

    print(un_shape)

    t = ContinuousTransformerWrapper(
        dim_in= 256,
        dim_out= 256,
        max_seq_len=1024,
        attn_layers = Decoder(
            dim = 256,
            depth = 1,
            heads = 1,
            alibi_pos_bias = True,
            alibi_num_heads = 1
        )
    ).cuda()

    embedder = KoopSim().cuda()

    cri = nn.MSELoss()
    optimizer = optim.Adam(embedder.parameters(), lr=0.0001)

    p = p.cuda()
    for epcoh in range(1):    
        for i in range(p.size(0)):
            x = p[0][i].unsqueeze(0)

            optimizer.zero_grad()

            e, d = embedder(x)

            loss = cri(d, x)
            loss.backward()
            optimizer.step()
    

    embedder.eval()
    one_sample = torch.stack([embedder(p[0][i].unsqueeze(0))[0] for i in range(p.size(1))], dim=0).cuda()


    cri = nn.MSELoss()
    optimizer = optim.Adam(t.parameters(), lr=0.0001)

    p = p.cuda()
    for epcoh in range(1): 
        print(t(one_sample))



    ## train koopman sim


    #for i in range(p.size(1)):
    #   plt.imshow(p[0][i][0], vmin=0, vmax=1, cmap=plt.cm.gray,
    #                    interpolation='bicubic', animated=True, origin='lower')
    #    plt.show()
