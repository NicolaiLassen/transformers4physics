import json
import h5py
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from config.config_autoregressive import AutoregressiveConfig
from config.config_emmbeding import EmmbedingConfig
from config.phys_config import PhysConfig
import torch
from time import time_ns

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
    date = '00'
    time = 'no dynamics'
    transformer_suffix = '_500'
    show_losses = True
    init_len = 1
    val_every_n_epoch = 50
    test_batch_sizes = []
    test_batch_sizes = [4,8,16,32,64,128]

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
    model = LandauLifshitzGilbertEmbedding(EmmbedingConfig(cfg)).cuda()
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

    warmup_transformer = autoregressive.generate(torch.rand((1,1,128)).cuda(),2)
    warmup_transformer = autoregressive.generate(torch.rand((1,1,128)).cuda(),25)
    warmup_embedder = model.embed(sample_t[0:1], field_t[0:1])
    warmup_embedder = model.recover(warmup_embedder)

    a = model.embed(sample_t, field_t)
    recon_a = model.recover(a)
    recon_a = recon_a.detach().cpu().numpy()

    time_embed_start = time_ns()
    init = model.embed(sample_t[0:init_len], field_t[0:init_len])
    time_embed_end = time_ns()
    # init = init.unsqueeze(0)
    # emb_seq = autoregressive.generate(init,max_length=400)
    time_transformer_start = time_ns()
    emb_seq = autoregressive.generate(init,seq_len=400-init_len)
    time_transformer_end = time_ns()
    batches_times = []
    for batch_size_test in test_batch_sizes:
        test_batch = init.unsqueeze(0).repeat((batch_size_test,1,1))
        time_transformer_test_start = time_ns()
        test_huge = autoregressive.generate(test_batch,seq_len=400-init_len)
        time_transformer_test_end = time_ns()
        batches_times.append((time_transformer_test_end - time_transformer_test_start) * 1e-9)
    for b,t in zip(test_batch_sizes, batches_times):
        print('Time for batch of {}: {} s'.format(b,t))
    # emb_seq = emb_seq[0][0]

    # a = model.embed(sample_t, field_t)
    time_recover_start = time_ns()
    recon = model.recover(emb_seq)
    time_recover_end = time_ns()
    recon = torch.cat([sample_t[0:init_len],recon],dim=0)
    recon = recon.detach().cpu().numpy()
    recon_x = np.mean(recon[:,0].reshape(sample.shape[0],-1), axis=1)
    crosses_zero = np.argmax(recon_x < 0)

    if show_losses:
        f = h5py.File(path + 'transformer_losses.h5', 'r')
        losses = np.array(f['train'])
        l = np.arange(len(losses))
        plt.plot(l,losses)
        plt.title('Training Loss')
        plt.yscale('log')
        plt.ylabel('loss', fontsize=32)
        plt.xlabel('epoch', fontsize=32)
        plt.grid()
        plt.show()

        if 'val' in f.keys():
            losses_val = np.array(f['val'])
            l = np.arange(val_every_n_epoch, (len(losses_val)+1)*val_every_n_epoch, val_every_n_epoch)
            f.close()
            plt.plot(l,losses_val)
            plt.title('Validation Loss')
            plt.yscale('log')
            plt.ylabel('loss_val', fontsize=32)
            plt.xlabel('epoch', fontsize=32)
            plt.grid()
            plt.show()

        f.close()

    width = 0.002
    headwidth = 2
    headlength = 5
    plt.quiver(sample[0,0].T, sample[0,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0), width=width, headwidth=headwidth, headlength=headlength)
    plt.quiver(recon[0,0].T, recon[0,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7), width=width, headwidth=headwidth, headlength=headlength)
    plt.axis("scaled")
    plt.show()
    plt.quiver(sample[-1,0].T, sample[-1,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0), width=width, headwidth=headwidth, headlength=headlength)
    plt.quiver(recon[-1,0].T, recon[-1,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7), width=width, headwidth=headwidth, headlength=headlength)
    plt.axis("scaled")
    plt.show()
    plt.quiver(recon[crosses_zero,0].T, recon[crosses_zero,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0), width=width, headwidth=headwidth, headlength=headlength)
    plt.axis("scaled")
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
    # plt.title('Compared to ground truth')
    legend_elements = [Line2D([0], [0], color='black', lw=4, label='MagTense'),
                   Line2D([0], [0], marker='x', color='black', label='Model')]
    plt.legend(handles=legend_elements)
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

    time_embed = (time_embed_end - time_embed_start) * 1e-9
    time_transformer = (time_transformer_end - time_transformer_start) * 1e-9
    time_recover = (time_recover_end - time_recover_start)* 1e-9
    print('Time to embed: {} s'.format(time_embed))
    print('Time to generate: {} s'.format(time_transformer))
    print('Time to recover: {} s'.format(time_recover))
    print('Total time: {} s'.format(time_embed + time_transformer + time_recover))