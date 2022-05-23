import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from config.config_emmbeding import EmmbedingConfig
from config.phys_config import PhysConfig
import torch

from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding
from embedding.embedding_landau_lifshitz_gilbert_embed_mag import LandauLifshitzGilbertEmbeddingEM
from embedding.embedding_landau_lifshitz_gilbert_ff import LandauLifshitzGilbertEmbeddingFF
from embedding.embedding_landau_lifshitz_gilbert_ss import LandauLifshitzGilbertEmbeddingSS

if __name__ == '__main__':
    base = 'C:\\Users\\s174270\\Documents\\datasets\\64x16 field'
    show_losses = True
    # date = '2022-05-16'
    # time = '11-19-55'
    # model_name = 'val_3'
    date = '2022-05-23'
    time = '13-47-42'
    model_name = 'val_4'

    # f = h5py.File(base + '\\field_s_state_test_large.h5')
    f = h5py.File('./problem4.h5')
    # f = h5py.File(base + '\\field_s_state_test_circ_paper.h5')

    with open('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\{}\\{}\\dataset.txt'.format(date,time)) as file:
        print(file.read(-1))
    sample = np.array(f['1']['sequence'])
    field = np.array( f['1']['field'])
    # print(sample.shape)
    # plt.quiver(sample[-1,0].T, sample[-1,1].T, pivot='mid')
    # plt.show()
    class Object(object):
        pass

    cfg = Object()
    with open('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\{}\\{}\\ckpt\\config.json'.format(date,time), 'r', encoding="utf-8") as file:
        cfg_str = file.read(-1)
        cfg_json = json.loads(cfg_str)
        file.close()
    cfg.backbone= cfg_json["backbone"]
    cfg.backbone_dim = cfg_json["backbone_dim"]
    cfg.channels= cfg_json["channels"]
    cfg.ckpt_path= cfg_json["ckpt_path"]
    cfg.config_name= cfg_json["config_name"]
    cfg.embedding_dim= cfg_json["embedding_dim"]
    cfg.fc_dim= cfg_json["fc_dim"]
    cfg.image_size_x= cfg_json["image_size_x"]
    cfg.image_size_y= cfg_json["image_size_y"]
    cfg.koopman_bandwidth= cfg_json["koopman_bandwidth"]
    cfg.use_koop_net = False if "use_koop_net" not in cfg_json else cfg_json["use_koop_net"]
    model = LandauLifshitzGilbertEmbeddingEM(
        EmmbedingConfig(cfg)
    ).cuda()
    model.load_model('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\{}\\{}\\ckpt\\{}.pth'.format(date,time,model_name))
    model.eval()
    # print(sample.shape)
    sample_t = torch.tensor(sample).float().cuda()
    field_t = torch.zeros((sample_t.size(0),3)).float().cuda()
    field_t[:] = torch.tensor(field)
    a = model.embed(sample_t, field_t)
    recon = model.recover(a)

    # normsX = torch.sqrt(torch.einsum('ij,ij->j',sample_t.swapaxes(1,3).reshape(-1,3).T, sample_t.swapaxes(1,3).reshape(-1,3).T))
    # normsX = normsX.reshape(-1,16,64).swapaxes(1,2)
    # sample_t[:,0,:,:] = sample_t[:,0,:,:]/normsX
    # sample_t[:,1,:,:] = sample_t[:,1,:,:]/normsX
    # sample_t[:,2,:,:] = sample_t[:,2,:,:]/normsX

    # normsG = torch.sqrt(torch.einsum('ij,ij->j',recon.swapaxes(1,3).reshape(-1,3).T, recon.swapaxes(1,3).reshape(-1,3).T))
    # normsG = normsG.reshape(-1,16,64).swapaxes(1,2)
    # recon[:,0,:,:] = recon[:,0,:,:]/normsG
    # recon[:,1,:,:] = recon[:,1,:,:]/normsG
    # recon[:,2,:,:] = recon[:,2,:,:]/normsG

    recon = recon.detach().cpu().numpy()

    if show_losses:
        f = h5py.File('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\{}\\{}\\losses.h5'.format(date,time), 'r')
        losses = np.array(f['train'])
        l = np.arange(len(losses))
        plt.plot(l,losses)
        plt.yscale('log')
        plt.show()
        losses = np.array(f['val'])
        l = np.arange(len(losses))
        plt.plot(l,losses)
        plt.yscale('log')
        plt.show()

    plt.quiver(sample[0,0].T, sample[0,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    plt.quiver(recon[0,0].T, recon[0,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    plt.show()
    plt.quiver(sample[-1,0].T, sample[-1,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    plt.quiver(recon[-1,0].T, recon[-1,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    plt.show()
    
    plt.plot(np.arange(sample.shape[0]), np.mean(sample[:,0].reshape(sample_t.size(0),-1), axis=1), 'r')
    plt.plot(np.arange(sample.shape[0]), np.mean(sample[:,1].reshape(sample_t.size(0),-1), axis=1), 'g')
    plt.plot(np.arange(sample.shape[0]), np.mean(sample[:,2].reshape(sample_t.size(0),-1), axis=1), 'b')
    # plt.grid()
    # plt.show()
    
    plt.plot(np.arange(sample.shape[0]), np.mean(recon[:,0].reshape(sample_t.size(0),-1), axis=1), 'rx')
    plt.plot(np.arange(sample.shape[0]), np.mean(recon[:,1].reshape(sample_t.size(0),-1), axis=1), 'gx')
    plt.plot(np.arange(sample.shape[0]), np.mean(recon[:,2].reshape(sample_t.size(0),-1), axis=1), 'bx')
    plt.grid()
    plt.show()