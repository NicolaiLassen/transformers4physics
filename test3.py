import torch
from x_transformers import TransformerWrapper, Decoder
import h5py
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt

f = h5py.File("C:\\Users\\nicol\\OneDrive\\Desktop\\master\\smoke\\solved_fluid.h5", "r")

xdata = f.get('1')
xdata= np.array(xdata)

def split_tensor(tensor, channels=1, tile_size=256):
    mask = torch.ones_like(tensor)
    # use torch.nn.Unfold
    stride  = tile_size//2
    unfold  = nn.Unfold(kernel_size=(tile_size, tile_size), stride=stride)
    # Apply to mask and original image
    mask_p  = unfold(mask)
    patches = unfold(tensor)
	
    patches = patches.reshape(channels, tile_size, tile_size, -1).permute(3, 0, 1, 2)
    if tensor.is_cuda:
        patches_base = torch.zeros(patches.size(), device=tensor.get_device())
    else: 
        patches_base = torch.zeros(patches.size())
	
    tiles = []
    for t in range(patches.size(0)):
         tiles.append(patches[[t], :, :, :])
    return tiles, mask_p, patches_base, (tensor.size(2), tensor.size(3))

tiles = split_tensor(torch.tensor(xdata)[0].unsqueeze(0), tile_size=64)[0]

print(len(tiles))
