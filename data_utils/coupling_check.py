import h5py
import numpy as np

if __name__ == '__main__':
    path = 'C:\\Users\\s174270\\Documents\\datasets\\64x16 field\\'
    file_name = 'material_train.h5'
    f = h5py.File(path + file_name, 'r')
    norms = np.zeros((50,400*64*16))
    for i in range(50):
        seq = f[str(i)]['sequence']
        seq = np.swapaxes(seq, 1, 3)
        seq = seq.reshape(400*16*64,3)
        norms[i] = np.sqrt(np.einsum('ij,ij->j', seq.T, seq.T))
    unique = np.unique(norms)
    print(unique)
    print(norms.flatten().shape)