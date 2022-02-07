# %%
import numpy as np
import random
import math

import os
import torch
import h5py
from magtense import magtense


# %%
def normalizeVector(vector):
    vector = np.array(vector)
    return vector/np.sqrt(np.sum(vector**2)), np.sqrt(np.sum(vector**2))

# generate simple prism grid to "sim" grains of micro


# TODO MAKE THIS TORCH
# TODO save as files nok create on go
def create_prism_grid(rows=2, columns=2, size=1, res=224):
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
                random.random(),
                random.random(),
                random.random(),
            ]
            ea, _ = normalizeVector(ea)
            tiles.set_easy_axis_i(ea, i)
            tiles.set_remanence_i(random.uniform(1.0, 1.5)/(4*math.pi*1e-7), i)
            tiles.set_M(tiles.u_ea[i]*tiles.M_rem[i], i)

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

    imageIn = torch.zeros((res, res, 4))
    mask = torch.zeros((res, res))
    for c in range(columns):
        for r in range(rows):
            i = r + c*rows
            normalizedM, lenM = normalizeVector(tiles.get_M(i))
            normalizedM, lenM = torch.tensor(normalizedM), torch.tensor(lenM)
            imageIn[
                startX+sideLen*r:startX+sideLen*(r+1),
                startY+sideLen*c:startY+sideLen*(c+1),
                0:3,
            ] = normalizedM
            imageIn[
                startX+sideLen*r:startX+sideLen*(r+1),
                startY+sideLen*c:startY+sideLen*(c+1),
                3,
            ] = lenM
            mask[
                startX+sideLen*r:startX+sideLen*(r+1),
                startY+sideLen*c:startY+sideLen*(c+1),
            ] = 1
    imageIn = torch.moveaxis(imageIn, 2, 0)

    pixelSize = size/sideLen
    pointStartX = pixelSize/2 - \
        (innerPadding if paddingDim == 1 else outerPadding)*pixelSize
    pointStartY = pixelSize/2 - \
        (innerPadding if paddingDim == 2 else outerPadding)*pixelSize
    points = torch.zeros((res*res, 3))
    for i in range(res):
        for j in range(res):
            points[j+i*res, :] = torch.tensor([pointStartX+j *
                                               pixelSize, pointStartY+i*pixelSize, 0])

    magtense.run_simulation(tiles, points)
    hField = magtense.get_H_field(tiles, points)

    imageOut = torch.zeros((res*res, 4))
    normalizedH = [normalizeVector(x)[0] for x in hField]
    lenH = [normalizeVector(x)[1] for x in hField]
    normalizedH, lenH = torch.tensor(normalizedH), torch.tensor(lenH)
    imageOut[:, 0:3] = normalizedH
    imageOut[:, 3] = lenH
    imageOut = imageOut.reshape((res, res, 4))
    imageOut = torch.moveaxis(imageOut, 2, 0)

    imageOut[3,:,:] = imageOut[3,:,:]*(4*math.pi*1e-7)
    imageIn[3,:,:] = imageIn[3,:,:]*(4*math.pi*1e-7)

    return imageIn, mask, imageOut


# %%


class PrismGridDataset(torch.utils.data.Dataset):
    def __init__(self, images_in, masks, images_target):
        self.images_in = images_in
        self.masks = masks
        self.images_target = images_target

    def __len__(self):
        return len(self.images_in)

    def __getitem__(self, idx):
        return self.images_in[idx], self.masks[idx], self.images_target[idx]


def create_dataset(set_size=1024, columns=[4], rows=[4], square_grid=False, res=224, size=1):
    images_in = []
    masks = []
    images_target = []
    for i in range(set_size):
        print('{:06d}/{:06d}'.format(i+1, set_size), end='\r')
        r = random.choice(rows)
        c = random.choice(columns) if square_grid == False else r
        image_in, mask, image_target = create_prism_grid(
            rows=r,
            columns=c,
            size=size,
            res=res,
        )
        images_in.append(image_in)
        masks.append(mask)
        images_target.append(image_target)

    return PrismGridDataset(images_in, mask, images_target)
