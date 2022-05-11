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

if __name__ == '__main__':

    to_patches = nn.Conv2d(1, 256, 8, stride = 4, padding = 3)
    print(to_patches(torch.rand(1, 256, 256)).shape)