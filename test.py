
from models.embedding.twins_svt_backbone import TwinsSVTBackbone


if __name__ == '__main__':
    import os
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import torch.optim as optim
    import torchvision.transforms as transforms
    import vit_pytorch.twins_svt
    from PIL import Image
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # img = Image.open(
    #   "C:\\Users\\nicol\\OneDrive\\Desktop\\master\\transformers4physics\\models\\embedding\\test.jpg")
    pil_to_tensor = torch.rand(1, 3, 32, 32).cuda()

    # wandb sweep sweep_embed.yaml
    # wandb sweep autoregressive.yaml
    model = TwinsSVTBackbone(img_dim=32).cuda()

    print(model(pil_to_tensor))

    exit()
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
