import torch
from torch import Tensor

class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    # log σ(X^T . Y))
    def log_sigmoid_probs(self, x: Tensor, y: Tensor) -> Tensor:
        # Z^T . HK
        x_t = x.transpose(1, 2)
        # Take the mean probability of x being y
        # batch channel length steps
        out = torch.einsum("blcs,bcls->blcs", x_t, y)
        # E
        out = torch.mean(out)
        out = torch.sigmoid(out)
        return torch.log(out)

    def forward(self, h_k: Tensor, z: Tensor, z_n: Tensor) -> Tensor:
        # - (log σ(Z^T . HK)) + λE(ZN~PN) [log σ(ZN^T . HK)])
        return - (self.log_sigmoid_probs(z, h_k) + self.log_sigmoid_probs(-z_n, h_k))