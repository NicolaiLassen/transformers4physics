from sqlalchemy import over
from re import M
from PIL import Image
from matplotlib import cm, pyplot as plt
import numpy as np
import seaborn as sns
import os


def showNorm(imageOut, mask=None):
    ax = sns.heatmap(imageOut[3], cmap="mako", mask=mask)
    ax.invert_yaxis()
    plt.show()


def showNorm3d(image, mask=None, override_channel=3):

    # # creating figures
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    # # setting color bar
    y,x,z = (np.ma.masked_array(image[override_channel], mask=mask)).nonzero()
    maskedVals = np.ma.masked_array(image[override_channel, :, :, :], mask=mask)
    maskedVals = maskedVals[~maskedVals.mask].flatten()
    cmap = sns.dark_palette("#69d", reverse=False, as_cmap=True)
    color_map = cm.ScalarMappable(cmap=cmap)
    color_map.set_array(maskedVals)
    s_min = 1
    s_max = 64
    s_vals = [s_min + (x - min(maskedVals)) * (s_max - s_min) / (max(maskedVals) - min(maskedVals)) for x in maskedVals]

    # # creating the heatmap
    if(np.all(maskedVals == maskedVals[0])):
        img = ax.scatter(x, y, z, marker='s', s=s_vals, color='Blue')
    else:
        img = ax.scatter(x, y, z, marker='s', s=s_vals, cmap=cmap, c=maskedVals)
        plt.colorbar(color_map)

    # # adding title and labels
    ax.set_title("3D Heatmap")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    max_range = np.array([x.max()-x.min(), y.max()-y.min(),
                          z.max()-z.min()]).max() / 2.0
    mean_x = x.mean()
    mean_y = y.mean()
    mean_z = z.mean()
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)

    # # displaying plot
    plt.show()
