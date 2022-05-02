import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from config.config_emmbeding import EmmbedingConfig
from config.phys_config import PhysConfig
import torch

from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding

if __name__ == '__main__':
    base = 'C:\\Users\\s174270\\Documents\\datasets\\64x16 field'
    f = h5py.File(base + '\\field_s_state_test.h5')
    sample = np.array(f['4']['sequence'])
    field = np.array( f['4']['field'])
    # print(sample.shape)
    # plt.quiver(sample[-1,0].T, sample[-1,1].T, pivot='mid')
    # plt.show()
    class Object(object):
        pass

    cfg = Object()
    cfg.state_dims = [2, 64, 128]
    cfg.input_dims = [2, 64, 128]
    cfg.backbone= "ResNet"
    cfg.backbone_dim = 160
    cfg.channels= 5
    cfg.ckpt_path= ""
    cfg.config_name= ""
    cfg.embedding_dim= 128
    cfg.fc_dim= 160
    cfg.image_size_x= 64
    cfg.image_size_y= 16
    cfg.koopman_bandwidth= 7
    model = LandauLifshitzGilbertEmbedding(
        EmmbedingConfig(cfg),
    ).cuda()
    model.load_model('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\2022-04-30\\15-47-40\\ckpt\\no_name.pth')
    model.eval()
    # print(sample.shape)
    sample_t = torch.tensor(sample).float().cuda()
    field_t = torch.zeros((sample_t.size(0),3)).float().cuda()
    field_t[:] = torch.tensor(field)
    a = model.embed(sample_t, field_t)
    recon = model.recover(a)
    recon = recon.detach().cpu().numpy()
    plt.quiver(sample[0,0].T, sample[0,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    plt.quiver(recon[0,0].T, recon[0,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    plt.show()
    plt.quiver(sample[-1,0].T, sample[-1,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    plt.quiver(recon[-1,0].T, recon[-1,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    plt.show()
    
    plt.plot(np.arange(sample.shape[0]), np.mean(sample[:,0].reshape(400,-1), axis=1), 'r')
    plt.plot(np.arange(sample.shape[0]), np.mean(sample[:,1].reshape(400,-1), axis=1), 'g')
    plt.plot(np.arange(sample.shape[0]), np.mean(sample[:,2].reshape(400,-1), axis=1), 'b')
    # plt.grid()
    # plt.show()
    
    plt.plot(np.arange(sample.shape[0]), np.mean(recon[:,0].reshape(400,-1), axis=1), 'rx')
    plt.plot(np.arange(sample.shape[0]), np.mean(recon[:,1].reshape(400,-1), axis=1), 'gx')
    plt.plot(np.arange(sample.shape[0]), np.mean(recon[:,2].reshape(400,-1), axis=1), 'bx')
    plt.grid()
    plt.show()