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
from torch import einsum, nn, unsqueeze
from x_transformers import (ContinuousAutoregressiveWrapper,
                            ContinuousTransformerWrapper, Decoder, Encoder)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def exists(val):
    return val is not None


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)

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

    def forward(self, qk_dots, c=2, r=2):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :j]

        bias_x = torch.fmod(torch.arange(c * r, device=device), r).to(device)
        bias_y = torch.arange(c).repeat_interleave(r).to(device)
        bias = torch.sqrt(bias_x**2 + bias_y**2) * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))

        self.register_buffer('bias', bias, persistent=False)
        return qk_dots + self.bias


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, alibi_num_heads=4, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.rel_pos = AlibiPositionalBias(alibi_num_heads)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        dots = self.rel_pos(dots)

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
        channels=3,
        encoder_max_seq_len=128,
        decoder_max_seq_len=32,
        patch_size=32,
        alibi_num_heads=2,
        heads=4,
        dim=512,
        dim_head=256,
        mlp_dim=512,
        dropout=0.
    ):
        super().__init__()

        self.encoder_max_seq_len = encoder_max_seq_len
        self.spatial_encoder = ContinuousTransformerWrapper(
            dim_in=256,
            dim_out=256,
            max_seq_len=encoder_max_seq_len,
            attn_layers= Encoder(
                dim=256,
                depth=2,
                heads=4,
                alibi_pos_bias=True,
                alibi_num_heads=2
            )
        )

        self.decoder_max_seq_len = decoder_max_seq_len
        self.autoregresive_decoder = ContinuousAutoregressiveWrapper(
            ContinuousTransformerWrapper(
                dim_in=256,
                dim_out=256,
                max_seq_len=decoder_max_seq_len,
                attn_layers=Decoder(
                    dim=256,
                    depth=2,
                    heads=4
                )
            ))

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)


        self.net.train(was_training)
        return start_tokens

    def forward(self, x):
        # x: b, s, p, d
        b, s, p, d, device = *x.shape, x.device
        ctx = self.decoder_max_seq_len
        
        p_e = torch.zeros((b, ctx, p, d), device=device)
        for i in range(ctx):
            # e: b, p, d
            e = self.spatial_encoder(x[:, i])
            p_e[:, i] = e
           
        # p: b, s, d
        pe = x[:, :ctx, :] + p_e
        acc_loss = 0
        for i in range(p):
            loss = self.autoregresive_decoder(pe[:, :ctx, i])
            acc_loss = acc_loss + loss

        return acc_loss

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
    patches_orig = patches_orig.view(
        patches.size(0), output_c, output_h, output_w)
    return patches_orig


class Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2,
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

            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2,
                               padding=1, output_padding=1, padding_mode="zeros"),
        )

    def encode(self, x):
        e = self.encoder(x)
        e = e.view(-1, 256*2*2)
        e = self.encoder_fc(e)
        return e

    def decode(self, e):
        d = self.decoder_fc(e)
        d = d.view(-1, 256, 2, 2)
        d = self.decoder(d)
        return d

    def forward(self, x):
        # encode
        e = self.encoder(x)
        e = e.view(-1, 256*2*2)
        e = self.encoder_fc(e)
        # decode
        d = self.decoder_fc(e)
        d = d.view(-1, 256, 2, 2)
        d = self.decoder(d)
        return e, d


if __name__ == '__main__':

    import torchvision.transforms.functional as TF
    from PIL import Image

    # to patches
    f = h5py.File(
        "C:\\Users\\nicol\\OneDrive\\Desktop\\master\\smoke\\solved_fluid.h5", "r")

    xdata = f.get('1')
    xdata = torch.tensor(np.array(xdata))[1:]

    p, un_shape = extract_patches(xdata.float(), 64, 1)

    embedder = Tokenizer().cuda()

    mse = nn.MSELoss()
    optimizer_embed = optim.Adam(embedder.parameters(), lr=0.0001)

    p = p.cuda()
    for epcoh in range(100):
        loss_acc= 0
        for i in range(p.size(1)):
            x = p[:, i]
            optimizer_embed.zero_grad()
            e, d = embedder(x)
            loss = mse(d, x)
            loss.backward()
            optimizer_embed.step()
            loss_acc = loss_acc + loss.item()
        print(loss_acc)

    embedder.eval()
    embed_p = torch.stack([embedder.encode(p[:, i]).detach()
                          for i in range(p.size(1))], dim=1).cuda()
    embed_p = embed_p.unsqueeze(0)

    model = SpatialFormer().cuda()
    optimizer_transformer = optim.Adam(model.parameters(), lr=0.0001)

    for i in range(100):
        optimizer_transformer.zero_grad()
        loss = model(embed_p)
        loss.backward()
        optimizer_transformer.step()
        print(loss)
