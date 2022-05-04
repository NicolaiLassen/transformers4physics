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
from transformer.phys_transformer_gpt2 import PhysformerGPT2


if __name__ == '__main__':
    base = 'C:\\Users\\s174270\\Documents\\datasets\\64x16 field'
    f = h5py.File(base + '\\field_s_state_test_large.h5')
    sample = np.array(f['7']['sequence'])
    field = np.array( f['7']['field'])
    date = '2022-05-05'
    time = '00-21-59'
    init_len = 1

    # plt.quiver(sample[-1,0].T, sample[-1,1].T, pivot='mid')
    # plt.show()
    class Object(object):
        pass

    cfg = Object()
    cfg.state_dims = [2, 64, 128]
    cfg.input_dims = [2, 64, 128]
    cfg.backbone= "ResNet"
    cfg.backbone_dim = 192
    cfg.channels= 5
    cfg.ckpt_path= ""
    cfg.config_name= ""
    cfg.embedding_dim= 128
    cfg.fc_dim= 192
    cfg.image_size_x= 64
    cfg.image_size_y= 16
    cfg.koopman_bandwidth= 7
    model = LandauLifshitzGilbertEmbedding(
        EmmbedingConfig(cfg),
    ).cuda()
    path = './transformer_output/{}/{}/'.format(date,time)
    model.load_model(path + 'embedder.pth'.format(date,time))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    with open(path + 'transformer_config.json', 'r') as file:
        transformer_cfg = json.loads(json.load(file))

    autoregressive = ContinuousTransformerWrapper(
        dim_in=128,
        dim_out=128,
        max_seq_len=transformer_cfg["ctx"],
        attn_layers=Decoder(
            dim=transformer_cfg["decoder_dim"],
            depth=transformer_cfg["depth"], 
            heads=transformer_cfg["heads"],
            macaron=False if "macaron" not in transformer_cfg else transformer_cfg["macaron"],
            shift_tokens = 0 if "shift_tokens" not in transformer_cfg else  transformer_cfg["shift_tokens"],
        ),
    ).cuda()
    autoregressive = ContinuousAutoregressiveWrapper(autoregressive)
    
    autoregressive.load_state_dict(torch.load(path + 'transformer.pth'.format(date,time)))
    autoregressive.eval()
    for p in autoregressive.parameters():
        p.requires_grad = False
    
    sample_t = torch.tensor(sample).float().cuda()
    field_t = torch.zeros((sample_t.size(0),3)).float().cuda()
    field_t[:] = torch.tensor(field)

    init = model.embed(sample_t[0:init_len], field_t[0:init_len])
    init = init.unsqueeze(0)
    # emb_seq = autoregressive.generate(init,max_length=400)
    emb_seq = autoregressive.generate(init,seq_len=400-init_len)
    emb_seq = torch.cat([init,emb_seq],dim=1)
    # emb_seq = emb_seq[0][0]

    # a = model.embed(sample_t, field_t)
    recon = model.recover(emb_seq)
    recon = recon.detach().cpu().numpy()



    f = h5py.File('./transformer_output/{}/{}/transformer_losses.h5'.format(date, time), 'r')
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
        l = np.arange(len(losses_val))
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
    