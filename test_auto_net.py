import h5py
import numpy as np
import matplotlib.pyplot as plt
from config.config_autoregressive import AutoregressiveConfig
from config.config_emmbeding import EmmbedingConfig
from config.phys_config import PhysConfig
import torch

from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding
from transformer.phys_transformer_gpt2 import PhysformerGPT2

if __name__ == '__main__':
    base = 'C:\\Users\\s174270\\Documents\\datasets\\64x16 field'
    f = h5py.File(base + '\\field_s_state.h5')
    sample = np.array(f['0']['sequence'])
    field = np.array( f['0']['field'])
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
    cfg.embedding_dim= 256
    cfg.fc_dim= 160
    cfg.image_size_x= 64
    cfg.image_size_y= 16
    cfg.koopman_bandwidth= 23
    model = LandauLifshitzGilbertEmbedding(
        EmmbedingConfig(cfg),
    ).cuda()
    model.load_model('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\2022-04-29\\16-21-50\\ckpt\\embedding.pth')
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    cfg_auto = Object()
    cfg_auto.activation_function = "gelu_new"
    cfg_auto.embedding_dim = 256
    cfg_auto.n_ctx = 16
    cfg_auto.n_layer = 6
    cfg_auto.n_head = 4
    cfg_auto.output_hidden_states = False
    cfg_auto.output_attentions = False

    autoregressive = PhysformerGPT2(AutoregressiveConfig(cfg_auto)).cuda()
    
    autoregressive.load_model('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\2022-04-29\\16-21-50\\ckpt\\transformer_model0.pth')
    autoregressive.eval()
    for p in autoregressive.parameters():
        p.requires_grad = False
    
    sample_t = torch.tensor(sample).float().cuda()
    field_t = torch.zeros((sample_t.size(0),3)).float().cuda()
    field_t[:] = torch.tensor(field)

    init = model.embed(sample_t[0:4], field_t[0:4])
    init = init.unsqueeze(0)
    emb_seq = autoregressive.generate(init,max_length=400)
    emb_seq = emb_seq[0][0]

    # a = model.embed(sample_t, field_t)
    recon = model.recover(emb_seq)
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