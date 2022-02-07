# %%
import numpy as np
import random
import math

import seaborn as sns
from matplotlib import pyplot as plt
import os
import torch
import h5py
from magtense import magtense

# %%


def normalizeVector(vector):
    vector = np.array(vector)
    return vector/np.sqrt(np.sum(vector**2)), np.sqrt(np.sum(vector**2))

# generate simple prism grid to "sim" grains of micro


# TODO save as files nok create on go
def create_prism_grid(rows=2, columns=2, size=1, res=224, uniform=False, plot=False):
    paddingDim = 0 if rows == columns else 1 if rows < columns else 2
    sideLen = min(res//rows, res//columns)
    if((res-sideLen*(rows if paddingDim == 2 else columns)) % 2 != 0):
        sideLen = sideLen - 1
    elif((res-sideLen*(columns if paddingDim == 2 else rows)) % 2 != 0):
        sideLen = sideLen - 1
    outerPadding = (res-sideLen*(rows if paddingDim == 2 else columns))//2
    innerPadding = (res-sideLen*(columns if paddingDim == 2 else rows))//2
    startX = innerPadding if paddingDim == 1 else outerPadding
    startY = innerPadding if paddingDim == 2 else outerPadding

    if(sideLen < 1):
        raise Exception('Image dimensions and rows/columns are not compatible')
    pixelSize = size/sideLen
    pointStartX = pixelSize/2 - \
        (innerPadding if paddingDim == 1 else outerPadding)*pixelSize
    pointStartY = pixelSize/2 - \
        (innerPadding if paddingDim == 2 else outerPadding)*pixelSize
    points = np.zeros((res, res, 3))
    for i in range(res):
        for j in range(res):
            points[i, j, :] = np.array([pointStartX+j *
                                            pixelSize, pointStartY+i*pixelSize, 0])
    points = points.reshape((res*res, 3))
    tiles = magtense.Tiles(rows*columns)
    tiles.set_tile_type(2)
    tiles.set_size([size, size, size])
    for c in range(columns):
        for r in range(rows):
            i = r+c*rows
            offset = [size/2+r*size, size/2+c*size, 0]
            tiles.set_offset_i(offset, i)
            tiles.set_center_pos_i(offset, i)
            ea = [
                random.random()*2-1,
                random.random()*2-1,
                random.random()*2-1,
            ] if not uniform else [
                1,
                0,
                0,
            ]
            ea, _ = normalizeVector(ea)
            tiles.set_easy_axis_i(ea, i)
            tiles.set_remanence_i(random.uniform(
                1.0, 1.5 if not uniform else 1.0)/(4*math.pi*1e-7), i)
            tiles.set_M(tiles.u_ea[i]*tiles.M_rem[i], i)
            
    magtense.run_simulation(tiles, points, console=False)
    hField = magtense.get_H_field(tiles, points)

    imageIn = np.zeros((res, res, 4))
    mask = np.zeros((res, res))
    for c in range(columns):
        for r in range(rows):
            i = r + c*rows
            normalizedM, lenM = normalizeVector(tiles.get_M(i))
            normalizedM, lenM = np.array(normalizedM), np.array(lenM)
            imageIn[
                startY+sideLen*c:startY+sideLen*(c+1),
                startX+sideLen*r:startX+sideLen*(r+1),
                0:3,
            ] = normalizedM
            imageIn[
                startY+sideLen*c:startY+sideLen*(c+1),
                startX+sideLen*r:startX+sideLen*(r+1),
                3,
            ] = lenM
            mask[
                startY+sideLen*c:startY+sideLen*(c+1),
                startX+sideLen*r:startX+sideLen*(r+1),
            ] = 1

    imageIn = np.moveaxis(imageIn, 2, 0)

    imageOut = np.zeros((res, res, 4))
    normalizedH = [normalizeVector(x)[0] for x in hField]
    lenH = [normalizeVector(x)[1] for x in hField]
    normalizedH, lenH = np.array(normalizedH), np.array(lenH)
    for i, (nh, lh) in enumerate(zip(normalizedH, lenH)):
        imageOut[i//res, i % res, 0:3] = nh
        imageOut[i//res, i % res, 3] = lh
    imageOut = np.moveaxis(imageOut, 2, 0)
    # Back to tesla
    imageOut[3, :, :] = imageOut[3, :, :]*(4*math.pi*1e-7)
    imageIn[3, :, :] = imageIn[3, :, :]*(4*math.pi*1e-7)

    if(plot):
        magtense.create_plot(tiles)

    return imageIn, mask, imageOut


# %%
# imgin, m, imgout = create_prism_grid(
#     rows=2,
#     columns=1,
#     res=4,
# )
# %%

class PrismGridDataset(torch.utils.data.Dataset):
    def __init__(self, images_in=None, masks=None, images_target=None):
        self.x = images_in
        self.m = masks
        self.y = images_target

    def open_hdf5(self, datapath):
        db = h5py.File(datapath, mode='r')
        self.x = torch.tensor(db['x']).float()
        self.m = torch.tensor(db['m']).float()
        self.y = torch.tensor(db['y']).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.m[idx], self.y[idx]

def create_dataset(set_size=512, columns=[4], rows=[4], square_grid=False, res=224, size=1, datapath="./data/prism_grid_dataset.hdf5"):
    images_in = np.zeros((set_size, 4, res, res))
    masks = np.zeros((set_size, res, res))
    images_target = np.zeros((set_size, 4, res, res))
    for i in range(set_size):
        print('{:06d}/{:06d}'.format(i+1, set_size), end="\r")
        r = random.choice(rows)
        c = random.choice(columns) if square_grid == False else r
        image_in, mask, image_target = create_prism_grid(
            rows=r,
            columns=c,
            size=size,
            res=res,
        )

        images_in[i] = image_in
        masks[i] = mask
        images_target[i] = image_target

    # TODO
    with h5py.File(datapath, "w") as f:
        f.create_dataset("x", data=images_in)
        f.create_dataset("m", data=masks)
        f.create_dataset("y", data=images_target)

    return PrismGridDataset(images_in, masks, images_target)
