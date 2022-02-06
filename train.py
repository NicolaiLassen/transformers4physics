import hydra
from util.magtense.prism_grid import create_prism_grid
from models.CoAtNet import CoAtED
import pytorch_lightning as pl
import torch


def coatnet_0():
    num_blocks = [2, 2, 3, 5, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtED((8, 8), 4, num_blocks, channels)

## TEST of single l -> t
def train(cfg):
    l, t = create_prism_grid(
        rows=1,
        columns=2,
        res=8
    )

    l = torch.from_numpy(l)
    t = torch.from_numpy(t)

    # model = N2H(cfg)

    print(l.shape)

    net = coatnet_0()

    net(l)

    m = torch.nn.Conv3d(4, 1, 244, stride=2)
    m(l)
    # out = model.forward(l)

# @hydra.main(config_path=".", config_name="./models/N2H_train.yaml")
def main():  # (cfg):
    # pl.seed_everything(cfg.seed)
    train({})


if __name__ == '__main__':
    main()
