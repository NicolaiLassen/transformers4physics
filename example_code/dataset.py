import os
import torch
import h5py

class HalbachCylinderDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, field_input, action_input, target):
        super(HalbachCylinderDataset, self).__init__()
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