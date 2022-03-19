
from statistics import mode
from config.config_emmbeding import EmmbedingConfig
from embeddings.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding, LandauLifshitzGilbertEmbeddingTrainer
from models.embedding.twins_svt_backbone import TwinsSVTBackbone


if __name__ == '__main__':
    import os
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import torch.optim as optim
    import torchvision.transforms as transforms
    from PIL import Image

    model = LandauLifshitzGilbertEmbeddingTrainer(
        EmmbedingConfig(
            image_dim=32,
            embedding_dim=128,
            fc_layer=128,
            pretrained=False
        )
    ).cuda()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # img = Image.open(
    #   "C:\\Users\\nicol\\OneDrive\\Desktop\\master\\transformers4physics\\models\\embedding\\test.jpg")
    pil_to_tensor = torch.rand(1, 10, 3, 32, 32).cuda()

    print(pil_to_tensor[0, 0, :].shape)
    plt.imshow(transforms.ToPILImage()(pil_to_tensor[0, 0, :].cpu()))
    plt.show()

    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    for i in range(1000):
        loss, loss_reconstruct = model(pil_to_tensor)

        loss0 = loss.sum()
        loss0.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()
        print(loss0)

        if i % 100 == 0:
            print(pil_to_tensor[:, 0, :, :].shape)
            g,r = model.embedding_model(pil_to_tensor[:, 0, :, :])
            plt.imshow(transforms.ToPILImage()(r[0].cpu()))
            plt.show()
            plt.axis('off')


    exit()

    # wandb sweep sweep_embed.yaml
    # wandb sweep autoregressive.yaml
    model = TwinsSVTBackbone(img_dim=32).cuda()
    criterion = nn.MSELoss()

    for i in range(10000):
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pil_to_tensor)
        loss = criterion(outputs, pil_to_tensor)
        loss.backward()
        optimizer.step()

        print(loss.item())

        if i % 100 == 0:
            plt.imshow(transforms.ToPILImage()(
                pil_to_tensor.cpu().squeeze_(0)))
            plt.show()
            plt.imshow(transforms.ToPILImage()(outputs.cpu().squeeze_(0)))
            plt.axis('off')
            plt.show()
