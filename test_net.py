import json
import h5py
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from config.config_emmbeding import EmmbedingConfig
from config.phys_config import PhysConfig
import torch

from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding
from embedding.embedding_landau_lifshitz_gilbert_ff import LandauLifshitzGilbertEmbeddingFF
from embedding.embedding_landau_lifshitz_gilbert_ss import LandauLifshitzGilbertEmbeddingSS

if __name__ == '__main__':
    base = 'C:\\Users\\s174270\\Documents\\datasets\\64x16 field'
    show_losses = True
    # date = '2022-05-16'
    # time = '11-19-55'
    # model_name = 'val_3'
    date = '00'
    time = 'no dynamics'
    model_name = 'val_5'
    val_every_n_epoch = 50

    # f = h5py.File(base + '\\field_s_state_test_large.h5')
    f = h5py.File('./problem4.h5')
    # f = h5py.File(base + '\\field_s_state_test_circ_paper.h5')

    with open('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\{}\\{}\\dataset.txt'.format(date,time)) as file:
        print(file.read(-1))
    sample = np.array(f['0']['sequence'])
    field = np.array( f['0']['field'])
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
    model = LandauLifshitzGilbertEmbedding(
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
        plt.title('Training Loss', fontsize=48)
        plt.yscale('log')
        plt.ylabel('Loss', fontsize=32)
        plt.xlabel('Epoch', fontsize=32)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid()
        plt.show()
        losses = np.array(f['val'])
        l = np.arange(len(losses))
        l = np.arange(val_every_n_epoch, (len(losses)+1)*val_every_n_epoch, val_every_n_epoch)
        plt.plot(l,losses)
        plt.title('Validation Loss', fontsize=48)
        plt.yscale('log')
        plt.ylabel('Loss', fontsize=32)
        plt.xlabel('Epoch', fontsize=32)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid()
        plt.show()

    plt.quiver(sample[0,0].T, sample[0,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    plt.quiver(recon[0,0].T, recon[0,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    plt.ylabel('y', fontsize=32, rotation = 0)
    plt.xlabel('x', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.show()
    plt.quiver(sample[-1,0].T, sample[-1,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    plt.quiver(recon[-1,0].T, recon[-1,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    plt.ylabel('y', fontsize=32, rotation = 0)
    plt.xlabel('x', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.show()

    timeline = np.arange(sample.shape[0]) * 4e-12 * 1e9
    
    plt.plot(timeline, np.mean(sample[:,0].reshape(sample_t.size(0),-1), axis=1), 'r')
    plt.plot(timeline, np.mean(sample[:,1].reshape(sample_t.size(0),-1), axis=1), 'g')
    plt.plot(timeline, np.mean(sample[:,2].reshape(sample_t.size(0),-1), axis=1), 'b')
    # plt.grid()
    # plt.show()
    
    plt.plot(timeline, np.mean(recon[:,0].reshape(sample_t.size(0),-1), axis=1), 'rx')
    plt.plot(timeline, np.mean(recon[:,1].reshape(sample_t.size(0),-1), axis=1), 'gx')
    plt.plot(timeline, np.mean(recon[:,2].reshape(sample_t.size(0),-1), axis=1), 'bx')
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='Mx MagTense'),
        Line2D([0], [0], color='green', lw=4, label='My MagTense'),
        Line2D([0], [0], color='blue', lw=4, label='Mz MagTense'),
        Line2D([0], [0], marker='x', color='red', label='Mx Model'),
        Line2D([0], [0], marker='x', color='green', label='My Model'),
        Line2D([0], [0], marker='x', color='blue', label='Mz Model'),
    ]
    plt.legend(handles=legend_elements)
    plt.ylabel('Spatially averaged magnetization', fontsize=32)
    plt.xlabel('Time (ns)', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid()
    plt.title('Auto-encoder', fontsize=48)
    plt.show()