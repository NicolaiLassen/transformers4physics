from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding
from config.config_emmbeding import EmmbedingConfig
import plotly.express as px
from sklearn.decomposition import PCA
import torch
import h5py
import numpy as np
import json

# TODO: 

n_seq = 20
label_1 = "Dynamics"
label_2 = "No dynamics"
file_path = "C:\\Users\\s174270\\Documents\\datasets\\64x16 field\\field_s_state_test_circ_paper.h5"

# read different data 
def read_n_data_points(embedder: LandauLifshitzGilbertEmbedding):
    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            data_series = torch.Tensor(np.array(f[key]['sequence']))
            field = torch.Tensor(np.array(f[key]['field'][:2])).unsqueeze(0)
            
            with torch.no_grad():
                embedded_series = embedder.embed(data_series, field).cpu()

            return embedded_series[0: n_seq]

# compare the models in same PCA space

class Object(object):
    pass

cfg_1 = Object()
with open('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\00\\circ paper net koop\\ckpt\\config.json', 'r', encoding="utf-8") as file:
    cfg_str = file.read(-1)
    cfg_json = json.loads(cfg_str)
    file.close()
cfg_1.backbone= cfg_json["backbone"]
cfg_1.backbone_dim = cfg_json["backbone_dim"]
cfg_1.channels= cfg_json["channels"]
cfg_1.ckpt_path= cfg_json["ckpt_path"]
cfg_1.config_name= cfg_json["config_name"]
cfg_1.embedding_dim= cfg_json["embedding_dim"]
cfg_1.fc_dim= cfg_json["fc_dim"]
cfg_1.image_size_x= cfg_json["image_size_x"]
cfg_1.image_size_y= cfg_json["image_size_y"]
cfg_1.koopman_bandwidth= cfg_json["koopman_bandwidth"]
cfg_1.use_koop_net = False if "use_koop_net" not in cfg_json else cfg_json["use_koop_net"]
model_1 = LandauLifshitzGilbertEmbedding(
    EmmbedingConfig(cfg_1)
).cuda()
model_1.load_model('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\00\\circ paper net koop\\ckpt\\val_4.pth')
model_1.eval()

# dynamics
x_1 = read_n_data_points(model_1)

cfg_2 = Object()
with open('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\00\\no dynamics\\ckpt\\config.json', 'r', encoding="utf-8") as file:
    cfg_str = file.read(-1)
    cfg_json = json.loads(cfg_str)
    file.close()
cfg_2.backbone= cfg_json["backbone"]
cfg_2.backbone_dim = cfg_json["backbone_dim"]
cfg_2.channels= cfg_json["channels"]
cfg_2.ckpt_path= cfg_json["ckpt_path"]
cfg_2.config_name= cfg_json["config_name"]
cfg_2.embedding_dim= cfg_json["embedding_dim"]
cfg_2.fc_dim= cfg_json["fc_dim"]
cfg_2.image_size_x= cfg_json["image_size_x"]
cfg_2.image_size_y= cfg_json["image_size_y"]
cfg_2.koopman_bandwidth= cfg_json["koopman_bandwidth"]
cfg_2.use_koop_net = False if "use_koop_net" not in cfg_json else cfg_json["use_koop_net"]
model_2 = LandauLifshitzGilbertEmbedding(
    EmmbedingConfig(cfg_2)
).cuda()
model_2.load_model('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\00\\no dynamics\\ckpt\\val_5.pth')
model_2.eval()

#  no dynamics
x_2 = read_n_data_points(model_2)

# label_1 + label_2
X = torch.cat([x_1, x_2], dim=0)

# group
color_map = ( [label_1] * n_seq) + ( [label_2] * n_seq) 

# plot
pca = PCA(n_components=3)
components = pca.fit_transform(X)

# var explained
total_var = pca.explained_variance_ratio_.sum() * 100

# plot
fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=color_map,
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig.show()