import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from config.config_autoregressive import AutoregressiveConfig
from config.config_emmbeding import EmmbedingConfig
from config.phys_config import PhysConfig
import torch

from x_transformers import ContinuousTransformerWrapper, Decoder
from x_transformers import ContinuousAutoregressiveWrapper
from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding
from embedding.embedding_landau_lifshitz_gilbert_ff import LandauLifshitzGilbertEmbeddingFF
from transformer.phys_transformer_gpt2 import PhysformerGPT2


if __name__ == '__main__':
    base = 'C:\\Users\\s174270\\Documents\\datasets\\64x16 field'
    # f = h5py.File(base + '\\field_s_state_test_large.h5')
    # f = h5py.File(base + '\\field_s_state_test_circ_paper.h5')
    f = h5py.File('./problem4.h5')
    sample_idx = 1
    sample = np.array(f[str(sample_idx)]['sequence'])
    field = np.array( f[str(sample_idx)]['field'])
    # date = '2022-05-06'
    # time = '22-20-04'
    date = '2022-05-24'
    time = '11-36-14'
    transformer_suffix = '_300'
    show_losses = True
    init_len = 1
    val_every_n_epoch = 75

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
    model = LandauLifshitzGilbertEmbeddingFF(EmmbedingConfig(cfg)).cuda()
    model.load_model(path + 'embedder.pth'.format(date,time))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


    autoregressive = ContinuousTransformerWrapper(
        dim_in=128 if "emb_size" not in transformer_cfg else transformer_cfg["emb_size"],
        dim_out=128 if "emb_size" not in transformer_cfg else transformer_cfg["emb_size"],
        max_seq_len=transformer_cfg["ctx"],
        attn_layers=Decoder(
            dim=transformer_cfg["decoder_dim"],
            depth=transformer_cfg["depth"], 
            heads=transformer_cfg["heads"],
            macaron=False if "macaron" not in transformer_cfg else transformer_cfg["macaron"],
            shift_tokens = 0 if "shift_tokens" not in transformer_cfg else transformer_cfg["shift_tokens"],
        ),
    ).cuda()
    autoregressive = ContinuousAutoregressiveWrapper(autoregressive)
    
    autoregressive.load_state_dict(torch.load(path + 'transformer{}.pth'.format(transformer_suffix)))
    autoregressive.eval()
    for p in autoregressive.parameters():
        p.requires_grad = False
    
    sample_t = torch.tensor(sample).float().cuda()
    field_t = torch.zeros((sample_t.size(0),3)).float().cuda()
    field_t[:] = torch.tensor(field)
    a = model.embed(sample_t, field_t)
    recon_a = model.recover(a)
    recon_a = recon_a.detach().cpu().numpy()

    init = model.embed(sample_t[0:init_len], field_t[0:init_len])
    # init = init.unsqueeze(0)
    # emb_seq = autoregressive.generate(init,max_length=400)
    emb_seq = autoregressive.generate(init,seq_len=400-init_len)
    emb_seq = torch.cat([init,emb_seq],dim=0)
    # emb_seq = emb_seq[0][0]

    # a = model.embed(sample_t, field_t)
    recon = model.recover(emb_seq)
    recon = recon.detach().cpu().numpy()

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
    plt.quiver(recon[0,0].T, recon[0,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    plt.show()
    plt.quiver(sample[-1,0].T, sample[-1,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    plt.quiver(recon[-1,0].T, recon[-1,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    plt.show()
    
    plt.plot(np.arange(sample.shape[0]), np.mean(sample[:,0].reshape(sample.shape[0],-1), axis=1), 'r')
    plt.plot(np.arange(sample.shape[0]), np.mean(sample[:,1].reshape(sample.shape[0],-1), axis=1), 'g')
    plt.plot(np.arange(sample.shape[0]), np.mean(sample[:,2].reshape(sample.shape[0],-1), axis=1), 'b')
    # plt.grid()
    # plt.show()
    
    plt.plot(np.arange(sample.shape[0]), np.mean(recon[:,0].reshape(sample.shape[0],-1), axis=1), 'rx')
    plt.plot(np.arange(sample.shape[0]), np.mean(recon[:,1].reshape(sample.shape[0],-1), axis=1), 'gx')
    plt.plot(np.arange(sample.shape[0]), np.mean(recon[:,2].reshape(sample.shape[0],-1), axis=1), 'bx')
    plt.grid()
    plt.title('Compared to ground truth')
    plt.show()

    plt.plot(np.arange(sample.shape[0]), np.mean(recon_a[:,0].reshape(sample.shape[0],-1), axis=1), 'r')
    plt.plot(np.arange(sample.shape[0]), np.mean(recon_a[:,1].reshape(sample.shape[0],-1), axis=1), 'g')
    plt.plot(np.arange(sample.shape[0]), np.mean(recon_a[:,2].reshape(sample.shape[0],-1), axis=1), 'b')
    # plt.grid()
    # plt.show()
    
    plt.plot(np.arange(sample.shape[0]), np.mean(recon[:,0].reshape(sample.shape[0],-1), axis=1), 'rx')
    plt.plot(np.arange(sample.shape[0]), np.mean(recon[:,1].reshape(sample.shape[0],-1), axis=1), 'gx')
    plt.plot(np.arange(sample.shape[0]), np.mean(recon[:,2].reshape(sample.shape[0],-1), axis=1), 'bx')
    plt.grid()
    plt.title('Compared to embed -> recon')
    plt.show()
    