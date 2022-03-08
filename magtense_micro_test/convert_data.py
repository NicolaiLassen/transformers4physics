import numpy as np
import h5py

a = np.load('magtense_micro_test\cube36_3d.npy').squeeze()
a = np.swapaxes(a, 1, 2)
a = a.reshape(500, 3, 36, 36)
a = np.swapaxes(a,2,3)
# shape = 500 3 36 36
# shape = seq C W  H

hf = h5py.File('./magtense_micro_test/cube36_3d.h5', 'w')
hf.create_dataset('dataset_1', data=a)
hf.close()