#%%
import numpy as np
import random
import math

import os
import torch
import h5py
from magtense import magtense


#%%
def normalizeVector(vector):
    vector = np.array(vector)
    return vector/np.sqrt(np.sum(vector**2)),np.sqrt(np.sum(vector**2))

# generate simple prism grid to "sim" grains of micro
def create_prism_grid(rows=2, columns=2, sizeX=1, sizeY=1, sizeZ=1, res=224):
    tiles = magtense.Tiles(rows*columns)
    tiles.set_tile_type(2)
    tiles.set_size([sizeX, sizeY, sizeZ])
    points = [[sizeX/2+x, sizeY/2+y, 0] for y in range(res) for x in range(res)]
    for c in range(columns):
        for r in range(rows):
            i = r+c*rows
            offset = [sizeX/2+r*sizeX,sizeY/2+c*sizeY,0]
            tiles.set_offset_i(offset,i)
            tiles.set_center_pos_i(offset,i)
            ea = [
                random.random(),
                random.random(),
                random.random(),
            ]
            ea,_ = normalizeVector(ea)
            tiles.set_easy_axis_i(ea,i)
            ## TODO Change remenance to a random value in a valid span.
            ## This value is taken from the magtense example
            ## https://github.com/cmt-dtu-energy/MagTense/blob/master/python/examples/validation_prism.py
            tiles.set_remanence_i(1.2/(4*math.pi*1e-7),i)
            tiles.set_M(tiles.u_ea[i]*tiles.M_rem[i],i)

    magtense.run_simulation(tiles,points)
    hField = magtense.get_H_field(tiles,points)

    paddingDim = 0 if rows==columns else 1 if rows < columns else 2
    sideLen = min(res//rows,res//columns)
    if((res-sideLen*(rows if paddingDim == 2 else columns))%2!=0):
        raise Exception('this setup can not be properly scaled (whole numbers for dimensions not possible)')
    elif((res-sideLen*(columns if paddingDim == 2 else rows))%2!=0):
        raise Exception('this setup can not be properly scaled (whole numbers for dimensions not possible)')
    outerPadding = (res-sideLen*(rows if paddingDim == 2 else columns))//2
    innerPadding = (res-sideLen*(columns if paddingDim == 2 else rows))//2
    startX = outerPadding + (innerPadding if paddingDim == 1 else 0)
    startY = outerPadding + (innerPadding if paddingDim == 2 else 0)

    imageIn = np.zeros((res,res,4))
    for c in range(columns):
        for r in range(rows):
            i = r + c*rows
            normalizedM, lenM = normalizeVector(tiles.get_M(i))

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

    imageOut = np.zeros((res*res,4))
    normalizedH = [normalizeVector(x)[0] for x in hField]
    lenH = [normalizeVector(x)[1] for x in hField]
    imageOut[:,0:3] = normalizedH
    imageOut[:,3] = lenH
    imageOut = imageOut.reshape((res,res,4))

    return imageIn, imageOut
#%%
class PrismGridDataset(torch.utils.data.Dataset):
    def __init__(self, images_in, images_target):
        self.images_in = images_in
        self.images_target = images_target

    def __len__(self):
        return len(self.images_in)

    def __getitem__(self, idx):
        return self.images_in[idx], self.images_target[idx]

def create_dataset(set_size=1024, columns=[4], rows=[4], square_grid=False, res=224, sizeX=1, sizeY=1, sizeZ=1):
    images_in = []
    images_target = []
    for _ in range(set_size):
        r = random.choice(rows)
        c = random.choice(columns) if square_grid == False else r
        image_in, image_target = create_prism_grid(
            rows=r,
            columns=c,
            sizeX=sizeX,
            sizeY=sizeY,
            sizeZ=sizeZ,
            res=res,
        )
        images_in.append(image_in)
        images_target.append(image_target)
    return PrismGridDataset(images_in, images_target)


#%%
a = create_dataset(
    set_size=5,
    columns=[4],
    rows=[4],
    res=224,
)

#%%
class PrismGridDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, field_input, action_input, target):
        super(PrismGridDataset, self).__init__()
        self.db_path = os.path.dirname(os.path.abspath(__file__)) + '/../data/' + datapath
        self.size = h5py.File(self.db_path, mode='r')['action'].shape[0]
        self.field_input = field_input
        self.action_input = action_input
        self.target = target

    def open_hdf5(self):
        self.db = h5py.File(self.db_path, mode='r')

    def __getitem__(self, idx):
        if not hasattr(self, 'db'):
            self.open_hdf5()
        
        field = self.db['field'][idx]
        symm_mat = self.db['soft_mat'][idx]
        action = self.db['action'][idx]
        action_pos = self.db['id_pos'][idx]
        p2p = self.db['p2p'][idx] * 1e6
        p2p_next = self.db['p2p_next'][idx] * 1e6
        sph_ampl = self.db['sph_harm'][idx]
        sph_ampl_next = self.db['sph_harm_next'][idx]
        
        field = torch.from_numpy(field.astype('float32'))
        p2p= torch.from_numpy(p2p.astype('float32'))
        p2p_next = torch.from_numpy(p2p_next.astype('float32'))
        
        # Field pre-processing
        if self.field_input == 'field':
            field_repr = field
        elif self.field_input == 'spherical_harmonics':
            field_repr = sph_ampl
        else:
            raise NotImplementedError()

        # Action pre-processing
        if self.action_input == 'index':
            a_input = action
        elif self.action_input == 'position':
            a_input = torch.from_numpy(action_pos.astype('float32'))
        elif self.action_input == 'one_hot':
            z = action[0] // 8
            y_dict = {0:1, 1:2, 2:0, 3:3, 4:0, 5:3, 6:1, 7:2}
            y = y_dict[action[0] % 8]
            x = (action[0] % 8) // 2
            a_input = (x, y, z)
        else:
            raise NotImplementedError()

        # Target pre-processing
        if self.target == 'p2p_next':
            target = p2p_next
        elif self.target == 'bins':
            p2p_diff = p2p - p2p_next
            if p2p_diff < -250:
                target = 0
            elif (p2p_diff > -250) and (p2p_diff <= -50):
                target = 1
            elif (p2p_diff > -50) and (p2p_diff <= 50):
                target = 2
            elif (p2p_diff > 50) and (p2p_diff <= 250):
                target = 3
            elif (p2p_diff > 250):
                target = 4

        return field_repr, a_input, target

    def __len__(self):
        return self.size

if __name__ == '__main__':
    create_prism_grid()

#%%