import torch
from torch.nn import functional as F
from models.conv3d_net import ConvMToH

from util.magtense.prism_grid_3d import create_prism_grid_3d
from util.magtense.prism_diagram import showNorm3d



if __name__ == '__main__':
    res = 8

    # model = SwinTransformer3D(
    #     in_chans=4,
    #     window_size=(4,4,4),

    # )
    model = ConvMToH(
        res=res,
    )
    model.cuda()

    model.train()
    x, mask, y = create_prism_grid_3d(4,4,2,seed=42,res=res,uniform_ea=[1,0,0],uniform_tesla=1.0)
    x = torch.tensor(x).float().unsqueeze(0).cuda()
    showNorm3d(y, res=4)
    y = torch.tensor(y).float().unsqueeze(0).cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # for i in range(401):
    #     y_hat = model(x)
    #     y_hat = y_hat.view(-1, 4, res, res, res)
    #     l = F.mse_loss(y_hat, y)
    #     l.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     print(l)
    #     if i%20==0:
    #         showNorm3d(y_hat.cpu().detach().numpy()[0], res=8)
