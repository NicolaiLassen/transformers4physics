import numpy as np
import h5py

a = np.load('magtense_micro_test\cube36_3d.npy').squeeze()
a = a.swapaxes(1,2).reshape(500,3,36,36).swapaxes(2,3)

hf = h5py.File('./magtense_micro_test/cube36_3d.h5', 'w')
hf.create_dataset('dataset_1', data=a)
hf.close()

b = np.load('magtense_micro_test\cube36_2d.npy').squeeze()
b = b.swapaxes(1,2).reshape(500,3,36,36).swapaxes(2,3)
hf = h5py.File('./magtense_micro_test/cube36_2d.h5', 'w')
hf.create_dataset('dataset_1', data=a)
hf.close()