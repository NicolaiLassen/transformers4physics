# %%
import numpy as np
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
def create_prism_grid_3d(
    rows=2,
    columns=2,
    depth=2,
    size=1,
    res=224,
    plot=False,
    console=False,
    seed=None,
    uniform_ea=None,
    uniform_tesla=None,
    restrict_z=False,
):
    rng = np.random.default_rng(seed)
    paddingDim = [
        1 if rows < columns or rows < depth else 0,
        1 if columns < rows or columns < depth else 0,
        1 if depth < rows or depth < columns else 0,
    ]
    sideLen = min(min(res//rows, res//columns), res//depth)
    if((res-sideLen*rows) % 2 != 0 or (res-sideLen*columns) % 2 != 0 or (res-sideLen*depth) % 2 != 0):
        sideLen = sideLen - 1
    if(sideLen < 1):
        raise Exception('Image dimensions and rows/columns are not compatible')
    padding = [
        (res-sideLen*rows)//2,
        (res-sideLen*columns)//2,
        (res-sideLen*depth)//2,
    ]

    pixelSize = size/sideLen
    pointStartX = pixelSize/2 - padding[0]*pixelSize
    pointStartY = pixelSize/2 - padding[1]*pixelSize
    pointStartZ = pixelSize/2 - padding[2]*pixelSize
    points = np.zeros((res, res, res, 3))
    for i in range(res):
        for j in range(res):
            for k in range(res):
                points[i, j, k, :] = np.array([pointStartX+j *
                                               pixelSize, pointStartY+i*pixelSize, pointStartZ+k*pixelSize])
    points = points.reshape((res*res*res, 3))
    tiles = magtense.Tiles(rows*columns*depth)
    tiles.set_tile_type(2)
    tiles.set_size([size, size, size])
    for d in range(depth):
        for c in range(columns):
            for r in range(rows):
                i = r+c*rows+d*rows*columns
                offset = [size/2+c*size, size/2+r*size, size/2+d*size]
                tiles.set_offset_i(offset, i)
                tiles.set_center_pos_i(offset, i)
                ea = [
                    rng.random()*2-1,
                    rng.random()*2-1,
                    rng.random()*2-1 if not restrict_z else 0,
                ] if not uniform_ea else uniform_ea
                ea, _ = normalizeVector(ea)
                tiles.set_easy_axis_i(ea, i)
                tiles.set_remanence_i(
                    ((rng.random()*0.5+1) if not uniform_tesla else uniform_tesla)/(4*math.pi*1e-7), i)
    _, hField = magtense.run_simulation(tiles, points, console=console)

    imageIn = np.zeros((res, res, res, 4))
    mask = np.ones((res, res, res))
    for d in range(depth):
        for c in range(columns):
            for r in range(rows):
                i = r + c*rows + d*rows*columns
                normalizedM, lenM = normalizeVector(tiles.get_M(i))
                normalizedM, lenM = np.array(normalizedM), np.array(lenM)
                imageIn[
                    padding[0]+sideLen*r:padding[0]+sideLen*(r+1),
                    padding[1]+sideLen*c:padding[1]+sideLen*(c+1),
                    padding[2]+sideLen*d:padding[2]+sideLen*(d+1),
                    0:3,
                ] = normalizedM
                imageIn[
                    padding[0]+sideLen*r:padding[0]+sideLen*(r+1),
                    padding[1]+sideLen*c:padding[1]+sideLen*(c+1),
                    padding[2]+sideLen*d:padding[2]+sideLen*(d+1),
                    3,
                ] = lenM
                mask[
                    padding[0]+sideLen*r:padding[0]+sideLen*(r+1),
                    padding[1]+sideLen*c:padding[1]+sideLen*(c+1),
                    padding[2]+sideLen*d:padding[2]+sideLen*(d+1),
                ] = 0
    imageIn = np.moveaxis(imageIn, 3, 0)

    imageOut = np.zeros((res, res, res, 4))
    normalizedH = [normalizeVector(x)[0] for x in hField]
    lenH = [normalizeVector(x)[1] for x in hField]
    normalizedH, lenH = np.array(normalizedH), np.array(lenH)
    for i, (nh, lh) in enumerate(zip(normalizedH, lenH)):
        imageOut[i//res//res, i//res % res, i % res, 0:3] = nh
        imageOut[i//res//res, i//res % res, i % res, 3] = lh
    # The target image is constructed correctly as the evaluation points are defined going from top to bottom
    imageOut = np.moveaxis(imageOut, 3, 0)
    # Back to tesla
    imageOut[3, :, :, :] = imageOut[3, :, :, :]*(4*math.pi*1e-7)
    imageIn[3, :, :, :] = imageIn[3, :, :, :]*(4*math.pi*1e-7)

    if(plot):
        magtense.create_plot(tiles)

    return imageIn, mask, imageOut
