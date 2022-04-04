import h5py
from viz import MicroMagViz
import torch
import matplotlib

matplotlib.use('qtagg')

hf = h5py.File('mag_data_with_field.h5', 'r')
print(len(hf.keys()))
print(hf['0']['field'][:])
print(hf['0']['sequence'].shape)
viz = MicroMagViz()
viz.plot_prediction(torch.tensor(hf['0']['sequence'][:]), torch.tensor(hf['0']['sequence'][:]),timescale=500*4e-12)