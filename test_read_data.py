import h5py
from config.config_emmbeding import EmmbedingConfig
from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding
from viz import MicroMagViz
import torch
import matplotlib

#matplotlib.use('qtagg')
import numpy as np

# x = torch.rand((1,3,64,16))
# f = torch.tensor([1,2,3])
# asd = x[:,:1]
# x = torch.cat([
#     x, 
#     f[0].unsqueeze(-1).unsqueeze(-1) * torch.ones_like(asd),
#     f[1].unsqueeze(-1).unsqueeze(-1) * torch.ones_like(asd),
#     f[2].unsqueeze(-1).unsqueeze(-1) * torch.ones_like(asd),
# ], dim=1)
# print(x[0,3])
# print(x[0,4])
# print(x[0,5])

# exit()
class Object(object):
    pass

cfg = Object()
cfg.backbone = 'TwinsSVT'
cfg.channels = 6
cfg.image_size = [64,16]
cfg.backbone_dim = 64
cfg.embedding_dim = 4
cfg.fc_dim = 64
field = torch.tensor([1.0,2.0,3.0])
abe = LandauLifshitzGilbertEmbedding(
    EmmbedingConfig(cfg),
)

x = torch.rand(1,3,64,16)
y = abe.embed(x, field)
yh = abe.koopman_operation(y, field)
xh = abe.recover(yh)

exit()
hf = h5py.File('mag_data_with_field.h5', 'r')
print(len(hf.keys()))
print(hf['0']['field'][:])
print(hf['0']['sequence'].shape)
viz = MicroMagViz()
viz.plot_prediction(torch.tensor(hf['0']['sequence'][:]), torch.tensor(hf['0']['sequence'][:]),timescale=500*4e-12)