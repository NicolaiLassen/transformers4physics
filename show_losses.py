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
losses_val_avg = []
for fname in fnames:
    f = h5py.File(fname, 'r')
    losses = np.array(f['train'])
    losses_avg.append(losses[-5:].mean())
    if 'val' in f.keys():
        losses_val = np.array(f['val'])
        losses_val_avg.append(losses_val[-5:].mean())
    else:
        losses_val_avg.append('Unknown')

losses_avg, losses_val_avg, fnames = zip(*sorted(zip(losses_avg, losses_val_avg, fnames)))
losses_avg, losses_val_avg, fnames = reversed(losses_avg), reversed(losses_val_avg), reversed(fnames)
for l,v,f in zip(losses_avg, losses_val_avg, fnames):
    print(f)
    print('Train: {}'.format(l))
    print('Val: {}'.format(v))
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