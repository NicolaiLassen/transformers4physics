import json
import h5py
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
    show_losses = False
    # date = '2022-05-14'
    # time = '10-51-15'
    # date = '2022-05-23'
    # time = '13-00-49'
    # model_name = 'val_1'
    date = '2022-05-23'
    time = '14-45-33'
    model_name = 'val_6'

    start_at = 0
    koop_forward = 1
    # f = h5py.File(base + '\\field_s_state_test_large.h5')
    # f = h5py.File('./problem4.h5')
    f = h5py.File(base + '\\field_s_state_test_circ_paper.h5')
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
    model = LandauLifshitzGilbertEmbeddingFF(
        EmmbedingConfig(cfg),
    ).cuda()
    # print(model.use_koop_net)
    model.load_model('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\{}\\{}\\ckpt\\{}.pth'.format(date,time,model_name))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # print(sample.shape)
    sample_t = torch.tensor(sample).float().cuda()[start_at].unsqueeze(0)
    field_t = torch.tensor(field[:2]).unsqueeze(0).cuda().float()
    a, _ = model(sample_t, field_t)
    b = torch.zeros(400-start_at,cfg.embedding_dim).cuda()
    c, _ = model(torch.tensor(sample[start_at:]).float().cuda()[koop_forward].unsqueeze(0), field_t)
    b[0] = a[0]
    for i in range(1,400-start_at):
        b[i] = model.koopman_operation(b[i-1].unsqueeze(0),field_t)[0]
    mse = torch.nn.MSELoss()
    print(mse(model.recover(c[0]),model.recover(b[koop_forward])))
    # b[start_at+koop_forward] = c[0]
    print(mse(sample_t, model.recover(model.embed(sample_t, field_t))))
    # print(mse(model.recover(sample_t[0]),model.recover(b[koop_to])))
    recon = model.recover(b)

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

    plt.quiver(sample[start_at,0].T, sample[start_at,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    plt.quiver(recon[start_at,0].T, recon[start_at,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    plt.show()
    plt.quiver(sample[koop_forward,0].T, sample[koop_forward,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    plt.quiver(recon[koop_forward,0].T, recon[koop_forward,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    plt.show()
    plt.quiver(sample[-1,0].T, sample[-1,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    plt.quiver(recon[-1,0].T, recon[-1,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    plt.show()
    
    plt.plot(np.arange(start_at, sample.shape[0]), np.mean(sample[start_at:,0].reshape(400-start_at,-1), axis=1), 'r')
    plt.plot(np.arange(start_at, sample.shape[0]), np.mean(sample[start_at:,1].reshape(400-start_at,-1), axis=1), 'g')
    plt.plot(np.arange(start_at, sample.shape[0]), np.mean(sample[start_at:,2].reshape(400-start_at,-1), axis=1), 'b')
    # plt.grid()
    # plt.show()
    
    plt.plot(np.arange(start_at, sample.shape[0]), np.mean(recon[:,0].reshape(400-start_at,-1), axis=1), 'rx')
    plt.plot(np.arange(start_at, sample.shape[0]), np.mean(recon[:,1].reshape(400-start_at,-1), axis=1), 'gx')
    plt.plot(np.arange(start_at, sample.shape[0]), np.mean(recon[:,2].reshape(400-start_at,-1), axis=1), 'bx')
    plt.grid()
    plt.show()