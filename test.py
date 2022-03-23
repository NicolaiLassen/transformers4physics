
from omegaconf import DictConfig

from config.config_emmbeding import EmmbedingConfig
from embeddings.embedding_landau_lifshitz_gilbert import \
    LandauLifshitzGilbertEmbeddingTrainer
from models.embedding.restnet_backbone import ResnetBackbone
from models.embedding.twins_svt_backbone import TwinsSVTBackbone

if __name__ == '__main__':
    import os

    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as transforms
    # with h5py.File(val_path, "r") as f:
    #     for key in f.keys():
    #         data_series = torch.Tensor(f[key])
    #         print(data_series)
    # exit()
    from PIL import Image

    # base_path = "C:\\Users\\s174270\\Documents\\datasets\\32x32 with field"
    # val_path = "{}\\val.h5".format(base_path)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # img = Image.open(
    #   "C:\\Users\\nicol\\OneDrive\\Desktop\\master\\transformers4physics\\models\\embedding\\test.jpg")
    pil_to_tensor = torch.rand(16, 500, 3, 32, 32).cuda()
    model = LandauLifshitzGilbertEmbeddingTrainer(
        EmmbedingConfig(DictConfig({
          "image_dim": 32,
          "channels": 3,
          "backbone": "ResNet",
          "fc_dim": 256,
          "embedding_dim": 140,
          "backbone_dim": 128,
        }))
    ).cuda()
        
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()

    print(sum(p.numel() for p in model.parameters()))

    loss1, _, _ = model.evaluate(pil_to_tensor)
    print(loss1)

