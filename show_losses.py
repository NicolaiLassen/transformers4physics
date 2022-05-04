import h5py
from matplotlib import pyplot as plt
import numpy as np


import os
from fnmatch import fnmatch

root = './transformer_output/'
pattern = "transformer_losses.h5"
fnames = []

for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            fnames.append(os.path.join(path, name))

losses_avg = []
for fname in fnames:
    f = h5py.File(fname, 'r')
    losses = np.array(f['train'])
    losses_avg.append(losses[-5:].mean())

losses_avg, fnames = zip(*sorted(zip(losses_avg, fnames)))
losses_avg, fnames = reversed(losses_avg), reversed(fnames)
for l,f in zip(losses_avg,fnames):
    print(f)
    print(l)
    print()



# l = np.arange(len(losses))
# plt.plot(l,losses)
# plt.yscale('log')
# plt.grid()
# plt.show()
# losses = np.array(f['val'])
# l = np.arange(len(losses))
# plt.plot(l,losses)
# plt.yscale('log')
# plt.grid()
# plt.show()