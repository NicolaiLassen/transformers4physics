import torch.nn as nn

# https://arxiv.org/abs/1710.05941
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return x * self.activation(x)
