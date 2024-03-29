from typing import List
from torch.nn import MultiheadAttention
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from embedding.backbone.recovery_net import RecoveryNet

from .embedding_backbone import EmbeddingBackbone


# https://www.sciencedirect.com/science/article/pii/S0304885319307978
# https://towardsdatascience.com/illustrated-10-cnn-architectures-95d78ace614d#e276


def _weights_init(m):
    """
        Initialization of CNN weights
    """
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    """
      Identity mapping between ResNet blocks with diffrenet size feature map
    """

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, backbone_dims):
        super(ResNet, self).__init__()

        self.in_planes = backbone_dims[0]

        self.layer1 = self._make_layer(
            block, backbone_dims[0], num_blocks[0], stride=2)
        self.layer2 = self._make_layer(
            block, backbone_dims[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            block, backbone_dims[2], num_blocks[2], stride=2)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)

        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

def resnetDown6(backbone_dims=[]):
    return ResNet(BasicBlock, [1, 1, 1], backbone_dims)


class ResnetBackbone(EmbeddingBackbone):
    model_name = "ResNet"

    def __init__(
        self,
        channels: int = 3,
        img_size: List[int] = [32, 32],
        backbone_dim: int = 128,
        embedding_dim: int = 128,
        fc_dim: int = 128
    ):
        super().__init__()

        print("Backbone: {}".format(self.model_name))

        final_patch_size = [img_size[0] // 8, img_size[1] // 8]
        self.final_patch_size = final_patch_size
        self.embedding_dim = embedding_dim
        self.backbone_dim = backbone_dim

        backbone_dims = [int(backbone_dim / 4),
                         int(backbone_dim / 2), backbone_dim]

        self.observable_net_layers = nn.Sequential(
            nn.Conv2d(channels, backbone_dims[0], kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(backbone_dims[0]),
            nn.ReLU(),
            resnetDown6(backbone_dims),
        )

        self.observable_net_fc_layers = nn.Sequential(
            nn.Linear(backbone_dim*self.final_patch_size[0]*self.final_patch_size[1], fc_dim),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(fc_dim, embedding_dim),
            nn.LayerNorm(embedding_dim, eps=1e-5),
        )

        self.recovery_net_fc_layers = nn.Sequential(
            nn.Linear(embedding_dim, fc_dim),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(fc_dim, backbone_dim*self.final_patch_size[0]*self.final_patch_size[1]),
            nn.LeakyReLU(0.02, inplace=True),
        )

        self.recovery_net_layers = RecoveryNet(backbone_dims).net

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

    def recover(self, x):
        out = self.recovery_net_fc(x)
        out = out.view(-1, self.backbone_dim,
                       self.final_patch_size[0], self.final_patch_size[1])
        out = self.recovery_net(out)
        return out

    def forward(self, x):
        out = self.embed(x)
        out = self.recover(out)
        return out


# if __name__ == '__main__':
#     # print(test(torch.rand(1, 16, 6, 6)).shape)
#     input_test = torch.rand(1, 3, 32, 32)
#     model = ResnetBackbone()
#     print(sum(p.numel() for p in model.parameters()))
#     print(model(input_test).shape)
