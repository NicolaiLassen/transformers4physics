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
losses_val_last = []
losses_val_lowest = []
for fname in fnames:
    f = h5py.File(fname, 'r')
    losses = np.array(f['train'])
    losses_avg.append(losses[-5:].mean())
    if 'val' in f.keys():
        losses_val = np.array(f['val'])
        losses_val_last.append(losses_val[-1])
        losses_val_lowest.append(min(losses_val))
    else:
        losses_val_last.append('Unknown')
        losses_val_lowest.append('Unknown')


losses_val_lowest, losses_val_last, losses_avg, fnames = zip(*sorted(zip(losses_val_lowest, losses_val_last, losses_avg, fnames)))
losses_avg, losses_val_last, losses_val_lowest, fnames = reversed(losses_avg), reversed(losses_val_last), reversed(losses_val_lowest), reversed(fnames)
for l,v,vl,f in zip(losses_avg, losses_val_last, losses_val_lowest, fnames):
    print(f)
    print('Train: {}'.format(l))
    print('Val last: {}'.format(v))
    print('Val lowest: {}'.format(vl))
    print()