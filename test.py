import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.swin_transformer import SwinTransformer

from util.magtense.prism_grid import PrismGridDataset, create_dataset

from matplotlib import pyplot as plt
import seaborn as sns; sns.set_theme()
def showNorm(imageOut):
    ax = sns.heatmap(imageOut[3], cmap="mako")
    ax.invert_yaxis()
    plt.show()

if __name__ == '__main__':
    ## TEST
    from torch.nn import functional as F

    dataset = PrismGridDataset()
    dataset.open_hdf5(
            "\\Users\\nicol\\OneDrive\\Desktop\\master\\transformers4physics\\data\\prism_grid_dataset_224.hdf5")

    net = SwinTransformer(in_chans=4, num_classes=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    for i in range(200):
        y_hat = net(dataset.x[:2])
        print(y_hat)
        l = F.mse_loss(y_hat, dataset.y[:2])
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(l)
        if i % 20 == 0:
            showNorm(y_hat[0].detach().numpy())

