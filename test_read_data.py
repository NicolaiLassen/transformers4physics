import h5py
from config.config_emmbeding import EmmbedingConfig
from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding, LandauLifshitzGilbertEmbeddingTrainer
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
cfg.backbone = 'ResNet'
cfg.channels = 5
cfg.image_size = [64,16]
cfg.backbone_dim = 32
cfg.embedding_dim = 32
cfg.fc_dim = 32
cfg.state_dims = [2, 64, 128]
cfg.input_dims = [2, 64, 128]
field = torch.tensor([[-1.0,2.0],[3.0,2.0]]).cuda()
abe = LandauLifshitzGilbertEmbedding(
    EmmbedingConfig(cfg),
).cuda()

x = torch.rand(2,3,64,16).cuda()
y = abe.embed(x, field)
yh = abe.koopman_operation(y, field)
xh = abe.recover(yh)
x = torch.rand(64,6,3,64,16).cuda()
field = torch.rand(64,2).cuda()
mu = torch.tensor(
    [
        torch.mean(x[:, :, 0]),
        torch.mean(x[:, :, 1]),
        torch.mean(x[:, :, 2]),
        torch.mean(field[:, 0]),
        torch.mean(field[:, 1]),
        # torch.mean(field[:, 2]),
    ]
).cuda()
std = torch.tensor(
    [
        torch.std(x[:, :, 0]),
        torch.std(x[:, :, 1]),
        torch.std(x[:, :, 2]),
        torch.std(field[:, 0]),
        torch.std(field[:, 1]),
        # torch.std(field[:, 2]),
    ]
).cuda()
abe.mu = mu
abe.std = std
testing = LandauLifshitzGilbertEmbeddingTrainer(
    abe,
).cuda()
print(x.shape)
print(field.shape)
loss, lossR = testing(x, field)
print(loss)
print(lossR)

exit()
hf = h5py.File('mag_data_with_field.h5', 'r')
print(len(hf.keys()))
print(hf['0']['field'][:])
print(hf['0']['sequence'].shape)
viz = MicroMagViz()
viz.plot_prediction(torch.tensor(hf['0']['sequence'][:]), torch.tensor(hf['0']['sequence'][:]),timescale=500*4e-12)