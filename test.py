
from omegaconf import DictConfig
from torch.nn import MultiheadAttention

from config.config_emmbeding import EmmbedingConfig
from embedding.embedding_landau_lifshitz_gilbert import \
    LandauLifshitzGilbertEmbedding, LandauLifshitzGilbertEmbeddingTrainer
if __name__ == '__main__':
    import os

    import h5py
    # import matplotlib.pyplot as plt
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
    pil_to_tensor = torch.rand(1, 16, 3, 32, 32).cuda()
    model_1 = LandauLifshitzGilbertEmbedding(
        EmmbedingConfig(DictConfig({
            "image_size": 32,
            "channels": 3,
            "backbone": "Conv",
            "fc_dim": 64,
            "embedding_dim": 64,
            "backbone_dim": 64,
        }))
    )

    trainer_1 = LandauLifshitzGilbertEmbeddingTrainer(model_1)

    optimizer = optim.Adam(trainer_1.parameters(), lr=0.001)

    print(sum(p.numel() for p in trainer_1.parameters()))

    for i in range(10):

        optimizer.zero_grad()

        loss, _ = trainer_1(pil_to_tensor)

        loss.backward()
        optimizer.step()

    print(loss)

    # model_2 = LandauLifshitzGilbertEmbeddingTrainer(
    #     EmmbedingConfig(DictConfig({
    #         "image_size": 32,
    #         "channels": 3,
    #         "backbone": "Conv",
    #         "fc_dim": 64,
    #         "embedding_dim": 64,
    #         "backbone_dim": 64,
    #     }))
    # ).cuda()

    # optimizer = optim.Adam(model_2.parameters(), lr=0.001)
    # print(sum(p.numel() for p in model_2.parameters()))

    # for i in range(10):

    #     optimizer.zero_grad()

    #     loss, _ = model_2(pil_to_tensor)

    #     loss.backward()
    #     optimizer.step()

    # print(loss)

    # model_3 = LandauLifshitzGilbertEmbeddingTrainer(
    #     EmmbedingConfig(DictConfig({
    #         "image_size": 32,
    #         "channels": 3,
    #         "backbone": "ResNet",
    #         "fc_dim": 64,
    #         "embedding_dim": 64,
    #         "backbone_dim": 64,
    #     }))
    # ).cuda()

    # optimizer = optim.Adam(model_3.parameters(), lr=0.001)
    # print(sum(p.numel() for p in model_3.parameters()))

    # for i in range(10):

    #     optimizer.zero_grad()

    #     loss, _ = model_3(pil_to_tensor)

    #     loss.backward()
    #     optimizer.step()

    # print(loss)
