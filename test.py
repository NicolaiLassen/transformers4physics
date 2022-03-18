
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
    )

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # img = Image.open(
    #   "C:\\Users\\nicol\\OneDrive\\Desktop\\master\\transformers4physics\\models\\embedding\\test.jpg")
    pil_to_tensor = torch.rand(1, 10, 3, 32, 32).cuda()
    
    print(model(pil_to_tensor))

    exit()

    # wandb sweep sweep_embed.yaml
    # wandb sweep autoregressive.yaml
    model = TwinsSVTBackbone(img_dim=32).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
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
