from abc import abstractmethod

import torch.nn as nn
from models.embedding.embedding_backbone import EmbeddingBackbone


class ConvBackbone(EmbeddingBackbone):
    model_name = "Conv"

    def __init__(
        self,
        channels: int = 3,
        img_size: int = 32,
        backbone_dim: int = 128,
        embedding_dim: int = 128,
        fc_dim: int = 128,
    ):
        super().__init__()

        print("Backbone: {}".format(self.model_name))

        final_patch_size = int(img_size / 8)
        self.final_patch_size = final_patch_size
        self.backbone_dim = backbone_dim
        self.embedding_dim = embedding_dim

        self.obsdim = embedding_dim

        backbone_dims = [int(backbone_dim / 4),
                         int(backbone_dim / 2), backbone_dim]

        self.observable_net_layers = nn.Sequential(
            nn.Conv2d(channels, backbone_dims[0], kernel_size=5, stride=2,
                      padding=2, padding_mode="zeros"),
            nn.BatchNorm2d(backbone_dims[0]),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(backbone_dims[0], backbone_dims[0], kernel_size=3, stride=1,
                      padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(backbone_dims[0]),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(backbone_dims[0], backbone_dims[1], kernel_size=3, stride=2,
                      padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(backbone_dims[1]),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(backbone_dims[1], backbone_dims[1], kernel_size=3, stride=1,
                      padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(backbone_dims[1]),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(backbone_dims[1], backbone_dims[2], kernel_size=3, stride=2,
                      padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(backbone_dims[2]),
            nn.LeakyReLU(0.02, inplace=True),
        )

        self.observable_net_fc_layers = nn.Sequential(
            nn.Linear(backbone_dims[2]*final_patch_size**2, fc_dim),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(fc_dim, embedding_dim),
            nn.LayerNorm(embedding_dim, eps=1e-5),
        )

        self.recovery_net_fc_layers = nn.Sequential(
            nn.Linear(embedding_dim, fc_dim),
            nn.LeakyReLU(1.0, inplace=True),
            nn.Linear(fc_dim, backbone_dim*final_patch_size**2),
            nn.LeakyReLU(0.02, inplace=True),
        )

        self.recovery_net_layers = nn.Sequential(
            nn.ConvTranspose2d(
                backbone_dims[2],  backbone_dims[1], kernel_size=3, stride=2, padding=1, padding_mode="zeros", output_padding=1
            ),
            nn.BatchNorm2d(backbone_dims[1]),
            nn.LeakyReLU(0.02, inplace=True),

            nn.ConvTranspose2d(
                backbone_dims[1], backbone_dims[0], kernel_size=3, stride=2, padding=1, padding_mode="zeros", output_padding=1
            ),
            nn.BatchNorm2d(backbone_dims[0]),
            nn.LeakyReLU(0.02, inplace=True),

            nn.ConvTranspose2d(
                backbone_dims[0], 3, kernel_size=3, stride=2, padding=1, padding_mode="zeros", output_padding=1
            ),
            nn.LeakyReLU(0.02, inplace=True)
        )

    def observable_net(self, x):
        return self.observable_net_layers(x)

    def observable_net_fc(self, x):
        return self.observable_net_fc_layers(x)

    def recovery_net(self, x):
        return self.recovery_net_layers(x)

    def recovery_net_fc(self, x):
        return self.recovery_net_fc_layers(x)

    def embed(self, x):
        out = self.observable_net(x)
        out = out.view(x.size(0), -1)
        out = self.observable_net_fc(out)
        return out

    @abstractmethod
    def recover(self, x):
        out = self.recovery_net_fc(x)
        out = out.view(-1, self.backbone_dim,
                       self.final_patch_size, self.final_patch_size)
        out = self.recovery_net(out)
        return out

    def forward(self, x):
        out = self.embed(x)
        out = self.recover(out)
        return out


# if __name__ == '__main__':
#     x = torch.rand((1, 3, 32, 32))
#     model = ConvBackbone()
#     model.s
#     print(model(x))
