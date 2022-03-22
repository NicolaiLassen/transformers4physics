from abc import abstractmethod
import torch.nn as nn
from models.embedding.embedding_backbone import EmbeddingBackbone


class Upscaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, padding_mode="zeros"
        )
        self.bnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.ConvTranspose2d(
            32, 3, kernel_size=3, stride=2, padding=1, padding_mode="zeros"
        )
        self.act = nn.LeakyReLU(0.02, inplace=True)

    def forward(self, x):
        x = self.conv1(x, output_size=(16, 16))
        x = self.bnorm1(x)
        x = self.act(x)
        x = self.conv2(x, output_size=(32, 32))
        return x


class ConvBackbone(EmbeddingBackbone):
    def __init__(
        self, channels=3, img_dim=32, backbone_dim=128, embedding_dim=128, fc_dim=128,
    ):
        super().__init__()
        self.channels = channels
        self.img_dim = img_dim
        self.backbone_dim = backbone_dim
        self.embedding_dim = embedding_dim
        self.fc_dim = fc_dim

        self.obsdim = embedding_dim

        self.observableNet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, padding_mode="zeros"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),
        )

        self.observableNetFC = nn.Sequential(
            nn.Linear(64 * 8 * 8, 8 * 8 * 8),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(8 * 8 * 8, embedding_dim),
            nn.LayerNorm(embedding_dim, eps=1e-5),
        )

        self.recoveryNetFc = nn.Sequential(
            nn.Linear(embedding_dim, 8 * 8 * 8),
            nn.LeakyReLU(1.0, inplace=True),
            nn.Linear(8 * 8 * 8, 64 * 8 * 8),
            nn.LeakyReLU(0.02, inplace=True),
        )

        self.recoveryNet = Upscaler()

    def observable_net(self, x):
        return self.observableNet(x)

    def observable_net_fc(self, x):
        return self.observableNetFC(x)

    def recovery_net(self, x):
        return self.recoveryNet(x)

    def recovery_net_fc(self, x):
        return self.recoveryNetFc(x)

    def embed(self, x):
        g0 = self.observableNet(x)
        g = self.observableNetFC(g0.view(g0.size(0), -1))
        return g

    @abstractmethod
    def recover(self, g):
        out = self.recoveryNetFc(g).view(-1, 64, 8, 8)
        out = self.recoveryNet(out)
        return out
