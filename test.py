import math
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
from torch import nn
from torchvision.transforms import Resize
from torch.nn.init import normal_

class HashSpatialPositionEmbeddings(nn.Module):
    """
    TODO
    """

    def __init__(self, patch_size, hse_grid_size, dim):
        super().__init__()

        self.patch_size = patch_size
        pos_emb_shape = (1, hse_grid_size * hse_grid_size, dim)
        self.position_embeddings = nn.Parameter(normal_(torch.empty(pos_emb_shape), std=0.02))

    def extract_image_patches(self, x, kernel, stride=1, dilation=1):
        # Do TF 'SAME' Padding
        print(x)
        b,c,h,w = x.shape
        h2 = math.ceil(h / stride)
        w2 = math.ceil(w / stride)
        pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
        pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
        x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
        
        # Extract patches
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        patches = patches.permute(0,4,5,1,2,3).contiguous()
        
        return patches.view(b,-1,patches.shape[-2], patches.shape[-1])

    def get_hashed_spatial_posisition_embedding_index(self):
        return torch.rand((1)) 

    def forward(self, x):
        x, m = self.extract_image_patches(x, 32, 32)
        print(x.shape)
        x_positions = self.get_hashed_spatial_posisition_embedding_index()
        x + torch.index_select(x[0], x_positions, axis=0)
        return x

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class MlpBlock(nn.Module):
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

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

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SpatialVIT(nn.Module):
    """
    Args:
        channels:
        patch_size: patch size.
        hse_grid_size: Hash-based positional embedding grid size.
        dim: 
    
        longer_side_lengths: List of longer-side lengths for each scale in the
        multi-scale representation.
        max_seq_len_from_original_res: Maximum number of patches extracted from
        original resolution. <0 means use all the patches from the original
        resolution. None means we don't use original resolution input.
    Returns:
        A concatenating vector of (patches, HSE, SCE, input mask). The tensor shape
        is (n_crops, num_patches, patch_size * patch_size * c + 3).
    """

    def __init__(
            self,
            *,
            dim,
            channels=3,
            patch_size,
            
            hse_grid_size,
            depth,
            heads,
            mlp_dim,
            pool='cls',
            dim_head=64,
            dropout=0.,
            emb_dropout=0.):

        super().__init__()

        # learnable
        self.to_spatial_embedding = \
            HashSpatialPositionEmbeddings(patch_size, hse_grid_size, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

    def forward(self, img):
        # extract_image_patches
        x = self.to_spatial_embedding(img)
        b, n, _ = x.shape

        exit(0)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.dropout(x)
        x = self.transformer(x)

        x = x[:, 0]

        return x


def split_tensor(tensor, tile_size=256):
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

if __name__ == '__main__':

    image = Image.open(
        'C:\\Users\\nicol\\OneDrive\\Desktop\\master\\transformers4physics\\dog.jpg')
    image = image.resize((128, 128), resample=0)
    x = TF.to_tensor(image)
    x = x.unsqueeze(0)

    v = SpatialVIT(
        patch_size=32,
        hse_grid_size=10,
        dim=224,
        depth=6,
        heads=16,
        mlp_dim=224,
        dropout=0.1,
        emb_dropout=0.1
    )

    out = split_tensor(x, 64)[0]
    print(len(out))
    # preds = v(x)
