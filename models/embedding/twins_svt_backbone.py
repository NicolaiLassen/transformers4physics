from abc import abstractmethod

import torch
import torch.nn as nn
from einops import rearrange
from torch import einsum, nn


class EmbeddingBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Save config in model

    @abstractmethod
    def observable_net(self, **kwargs):
        pass

    @abstractmethod
    def observable_net_fc(self, **kwargs):
        pass

    @abstractmethod
    def recovery_net(self, **kwargs):
        pass

    @abstractmethod
    def recovery_net_fc(self, **kwargs):
        pass


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PatchEmbedding(nn.Module):
    def __init__(self, *, in_channels, out_channels, patch_size):
        super().__init__()
        self.dim = in_channels
        self.dim_out = out_channels
        self.patch_size = patch_size
        self.proj = nn.Conv2d(patch_size ** 2 * in_channels, out_channels, 1)

    def forward(self, fmap):
        p = self.patch_size
        fmap = rearrange(
            fmap, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1=p, p2=p)
        fmap = self.proj(fmap)
        return fmap

class PatchExpansion(nn.Module):
    def __init__(self, *, in_channels, out_channels, patch_size):
        super().__init__()
        self.dim = in_channels
        self.dim_out = out_channels
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(int(in_channels / (patch_size ** 2)), out_channels, 1)

    def forward(self, fmap):
        p = self.patch_size
        fmap = rearrange(
            fmap, 'b (c p1 p2) h w -> b c (h p1) (w p2)', p1=p, p2=p)
        fmap = self.proj(fmap)
        return fmap

class PEG(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.proj = Residual(nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=in_channels, stride=1))

    def forward(self, x):
        return self.proj(x)

class LocalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., patch_size=7):
        super().__init__()
        inner_dim = dim_head * heads
        self.patch_size = patch_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, fmap):
        shape, p = fmap.shape, self.patch_size
        b, n, x, y, h = *shape, self.heads
        x, y = map(lambda t: t // p, (x, y))

        fmap = rearrange(
            fmap, 'b c (x p1) (y p2) -> (b x y) c p1 p2', p1=p, p2=p)

        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(
            t, 'b (h d) p1 p2 -> (b h) (p1 p2) d', h=h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim=- 1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(
            out, '(b x y h) (p1 p2) d -> b (h d) (x p1) (y p2)', h=h, x=x, y=y, p1=p, p2=p)
        return self.to_out(out)

class GlobalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., k=7):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, k, stride=k, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))

        q, k, v = map(lambda t: rearrange(
            t, 'b (h d) x y -> (b h) (x y) d', h=h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, y=y)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, in_channels, depth, heads=8, dim_head=64, mlp_mult=4, local_patch_size=7, global_k=7, dropout=0., has_local=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(in_channels, LocalAttention(in_channels, heads=heads, dim_head=dim_head,
                         dropout=dropout, patch_size=local_patch_size))) if has_local else nn.Identity(),
                Residual(PreNorm(in_channels, FeedForward(in_channels, mlp_mult,
                         dropout=dropout))) if has_local else nn.Identity(),
                Residual(PreNorm(in_channels, GlobalAttention(in_channels, heads=heads,
                         dim_head=dim_head, dropout=dropout, k=global_k))),
                Residual(PreNorm(in_channels, FeedForward(
                    in_channels, mlp_mult, dropout=dropout)))
            ]))

    def forward(self, x):
        for local_attn, ff1, global_attn, ff2 in self.layers:
            x = local_attn(x)
            x = ff1(x)
            x = global_attn(x)
            x = ff2(x)
        return x

class TwinsSVTBackbone(nn.Module):
    def __init__(
        self,
        channels=3,
        img_dim=32,
        embedding_dim=128,
        fc_layer=512
    ):
        super().__init__()
        
        final_patch_size = int(img_dim / 4)
        self.final_patch_size = final_patch_size
        self.embedding_dim = embedding_dim

        # Observable net
        # TODO: can be a for loop
        ## TODO local last
        observable_net_fc_layers = []
        observable_net_fc_layers.append(nn.Sequential(
            PatchEmbedding(in_channels = channels, out_channels=embedding_dim, patch_size=4),
            Transformer(in_channels = embedding_dim, depth = 1, local_patch_size = 4, global_k = 7, dropout = 0, has_local = True),
            PEG(in_channels = embedding_dim, kernel_size = 3),
            nn.Sigmoid()
        ))

        self.observable_net_fc_layers = nn.Sequential(
            nn.Linear(embedding_dim*final_patch_size**2, fc_layer),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(fc_layer, embedding_dim),
        )

        self.observable_net_layers = nn.Sequential(*observable_net_fc_layers)

        # Recovery net 
        self.recovery_net_fc_layers = nn.Sequential(
            nn.Linear(embedding_dim, fc_layer),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(fc_layer, embedding_dim*final_patch_size**2),
        )
        
        # TODO: can be a for loop
        ## TODO local last
        recovery_net_layers = []
        recovery_net_layers.append(nn.Sequential(
            ## TODO local last
            Transformer(in_channels = embedding_dim, depth = 1, local_patch_size = 4, global_k = 7, dropout = 0, has_local = True),
            PEG(in_channels = embedding_dim, kernel_size = 3),
            PatchExpansion(in_channels = embedding_dim, out_channels=3, patch_size=4),
            nn.Sigmoid()
        ))

        self.recovery_net_layers = nn.Sequential(*recovery_net_layers)
    
    def observable_net(self, x):
        return self.observable_net_layers(x)

    def observable_net_fc(self, x):
        return self.observable_net_fc_layers(x)

    def recovery_net(self, x):
        return self.recovery_net_layers(x)

    def recovery_net_fc(self, x):
        return self.recovery_net_fc_layers(x)

    def embed(self, x):
        out = self.observable_net(x)
        out = out.view(-1, self.embedding_dim*self.final_patch_size**2)
        out = self.observable_net_fc(out)
        return out

    def recover(self, x):
        out = self.recovery_net_fc(x)
        out = out.view(-1, self.embedding_dim, self.final_patch_size, self.final_patch_size)
        out = self.recovery_net(out)
        return out

    def forward(self, x):
        out = self.embed(x)
        out = self.recover(out)
        return out

if __name__ == '__main__':    
    import os

    import matplotlib.pyplot as plt
    import torch.optim as optim
    import torchvision.transforms as transforms
    import vit_pytorch.twins_svt
    from PIL import Image
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    img = Image.open("C:\\Users\\nicol\\OneDrive\\Desktop\\master\\transformers4physics\\models\\embedding\\test.jpg")
    pil_to_tensor = torch.rand(1, 3, 64, 64).cuda()

    # wandb sweep sweep_embed.yaml
    # wandb sweep autoregressive.yaml
    model = TwinsSVTBackbone(img_dim=64).cuda()

    print(model(pil_to_tensor).shape)
    
    print(sum(p.numel() for p in model.parameters()))
    exit()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()

    for i in range(10000):
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pil_to_tensor)
        loss = criterion(outputs, pil_to_tensor)
        loss.backward()
        optimizer.step()

        print(loss.item())
        
        if i % 100 == 0:
            plt.imshow(transforms.ToPILImage()(pil_to_tensor.cpu().squeeze_(0)))
            plt.show()
            plt.imshow(transforms.ToPILImage()(outputs.cpu().squeeze_(0)))
            plt.axis('off')
            plt.show()
