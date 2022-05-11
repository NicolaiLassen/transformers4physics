import h5py
from matplotlib import pyplot as plt
import numpy as np

f = h5py.File('losses.h5', 'r')
losses = np.array(f['train'])
l = np.arange(len(losses))
plt.plot(l,losses)
plt.yscale('log')
plt.show()
losses = np.array(f['val'])
l = np.arange(len(losses))
plt.plot(l,losses)
plt.yscale('log')
plt.show()