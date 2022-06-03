import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from config.config_autoregressive import AutoregressiveConfig
from config.config_emmbeding import EmmbedingConfig
from config.phys_config import PhysConfig
import torch
from transformer.all_in_one import AllInOne

from x_transformers import ContinuousTransformerWrapper, Decoder
from x_transformers import ContinuousAutoregressiveWrapper
from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding
from embedding.embedding_landau_lifshitz_gilbert_ff import LandauLifshitzGilbertEmbeddingFF
from transformer.phys_transformer_gpt2 import PhysformerGPT2


if __name__ == '__main__':
    base = 'C:\\Users\\s174270\\Documents\\datasets\\64x16 field'
    # f = h5py.File(base + '\\field_s_state_test_large.h5')
    f = h5py.File('./problem4.h5')
    sample_idx = 1
    sample = np.array(f[str(sample_idx)]['sequence'])
    field = np.array( f[str(sample_idx)]['field'])
    # date = '2022-05-06'
    # time = '22-20-04'
    date = '2022-06-03'
    time = '12-02-42'
    transformer_suffix = '_150'
    show_losses = True
    init_len = 1
    val_every_n_epoch = 25

    path = './transformer_output/{}/{}/'.format(date,time)
    with open(path + 'transformer_config.json', 'r') as file:
        transformer_cfg = json.loads(json.load(file))

    class Object(object):
        pass

    cfg = Object()
    with open(path + 'embedder_cfg.json', 'r', encoding="utf-8") as file:
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
    cfg.pretrained = False

    model = AllInOne(transformer_cfg['emb_size'], transformer_cfg, cfg).cuda()
    
    model.load_state_dict(torch.load(path + 'transformer{}.pth'.format(transformer_suffix)))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    with torch.no_grad():
        sample_t = torch.tensor(sample).float().cuda()[:init_len].unsqueeze(0)
        field_t = torch.tensor(field).float().cuda().unsqueeze(0)
        rest = model.generate(sample_t, field_t, 400-init_len)
        out = torch.cat((sample_t, rest), dim=1)
    out = out.squeeze(0).detach().cpu().numpy()

    if show_losses:
        f = h5py.File(path + 'transformer_losses.h5', 'r')
        losses = np.array(f['train'])
        l = np.arange(len(losses))
        plt.plot(l,losses)
        plt.title('Training Loss')
        plt.yscale('log')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid()
        plt.show()

        if 'val' in f.keys():
            losses_val = np.array(f['val'])
            l = np.arange(val_every_n_epoch, (len(losses_val)+1)*val_every_n_epoch, val_every_n_epoch)
            f.close()
            plt.plot(l,losses_val)
            plt.title('Validation Loss')
            plt.yscale('log')
            plt.ylabel('loss_val')
            plt.xlabel('epoch')
            plt.grid()
            plt.show()

        f.close()

    plt.quiver(sample[0,0].T, sample[0,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    plt.quiver(out[0,0].T, out[0,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    plt.show()
    plt.quiver(sample[-1,0].T, sample[-1,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    plt.quiver(out[-1,0].T, out[-1,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    plt.show()
    
    plt.plot(np.arange(sample.shape[0]), np.mean(sample[:,0].reshape(sample.shape[0],-1), axis=1), 'r')
    plt.plot(np.arange(sample.shape[0]), np.mean(sample[:,1].reshape(sample.shape[0],-1), axis=1), 'g')
    plt.plot(np.arange(sample.shape[0]), np.mean(sample[:,2].reshape(sample.shape[0],-1), axis=1), 'b')
    # plt.grid()
    # plt.show()
    
    plt.plot(np.arange(sample.shape[0]), np.mean(out[:,0].reshape(sample.shape[0],-1), axis=1), 'rx')
    plt.plot(np.arange(sample.shape[0]), np.mean(out[:,1].reshape(sample.shape[0],-1), axis=1), 'gx')
    plt.plot(np.arange(sample.shape[0]), np.mean(out[:,2].reshape(sample.shape[0],-1), axis=1), 'bx')
    plt.grid()
    plt.title('Compared to ground truth')
    plt.show()