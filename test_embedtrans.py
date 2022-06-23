import json
from time import time_ns
import h5py
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from config.config_autoregressive import AutoregressiveConfig
from config.config_emmbeding import EmmbedingConfig
from config.phys_config import PhysConfig
import torch
from matplotlib.pyplot import figure
from transformer.all_in_one import AllInOne

from x_transformers import ContinuousTransformerWrapper, Decoder
from x_transformers import ContinuousAutoregressiveWrapper
from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding
from embedding.embedding_landau_lifshitz_gilbert_ff import LandauLifshitzGilbertEmbeddingFF
from transformer.phys_transformer_gpt2 import PhysformerGPT2

def runit(sample_idx):
    base = 'C:\\Users\\s174270\\Documents\\datasets\\64x16 field'
    # f = h5py.File(base + '\\field_s_state_test_large.h5')
    f = h5py.File('./problem4.h5')
    sample = np.array(f[str(sample_idx)]['sequence'])
    field = np.array( f[str(sample_idx)]['field'])
    # date = '2022-05-06'
    # time = '22-20-04'
    date = '2022-06-03'
    time = '15-20-54'
    transformer_suffix = '_175'
    show_losses = False
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
        time_transformer_start = time_ns()
        rest = model.generate(sample_t, field_t, 400-init_len)
        time_transformer_end = time_ns()
        out = torch.cat((sample_t, rest), dim=1)
        recon_only = model.embed(torch.tensor(sample).float().cuda().unsqueeze(0), field_t)
        recon_only_out = model.recover(recon_only).squeeze(0).detach().cpu().numpy()
    out = out.squeeze(0).detach().cpu().numpy()
    print(out.shape)
    print(recon_only_out.shape)

    recon = out
    recon_x = np.mean(recon[:,0].reshape(sample.shape[0],-1), axis=1)
    crosses_zero = np.argmax(recon_x < 0)

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

    width = 0.002
    headwidth = 2
    headlength = 5
    # plt.quiver(sample[0,0].T, sample[0,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0), width=width, headwidth=headwidth, headlength=headlength)
    # plt.quiver(recon[0,0].T, recon[0,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7), width=width, headwidth=headwidth, headlength=headlength)
    # plt.axis("scaled")
    # plt.show()
    # plt.quiver(sample[-1,0].T, sample[-1,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0), width=width, headwidth=headwidth, headlength=headlength)
    # plt.quiver(recon[-1,0].T, recon[-1,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7), width=width, headwidth=headwidth, headlength=headlength)
    # plt.axis("scaled")
    # plt.show()
    figure(figsize=(16,8), dpi=140)
    plt.quiver(recon[crosses_zero,0].T, recon[crosses_zero,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0), width=width, headwidth=headwidth, headlength=headlength)
    plt.axis("scaled")
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\same time\\problem {} cross x0.png'.format(sample_idx + 1), format='png', bbox_inches='tight')
    
    timeline = np.arange(sample.shape[0]) * 4e-12 * 1e9

    figure(figsize=(16,9), dpi=140)
    plt.plot(timeline, np.mean(sample[:,0].reshape(sample.shape[0],-1), axis=1), 'r')
    plt.plot(timeline, np.mean(sample[:,1].reshape(sample.shape[0],-1), axis=1), 'g')
    plt.plot(timeline, np.mean(sample[:,2].reshape(sample.shape[0],-1), axis=1), 'b')
    # plt.grid()
    # plt.show()
    
    plt.plot(timeline, np.mean(recon[:,0].reshape(sample.shape[0],-1), axis=1), 'rx')
    plt.plot(timeline, np.mean(recon[:,1].reshape(sample.shape[0],-1), axis=1), 'gx')
    plt.plot(timeline, np.mean(recon[:,2].reshape(sample.shape[0],-1), axis=1), 'bx')
    # plt.title('Compared to ground truth')
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='Mx MagTense'),
        Line2D([0], [0], color='green', lw=4, label='My MagTense'),
        Line2D([0], [0], color='blue', lw=4, label='Mz MagTense'),
        Line2D([0], [0], marker='x', color='red', label='Mx Model'),
        Line2D([0], [0], marker='x', color='green', label='My Model'),
        Line2D([0], [0], marker='x', color='blue', label='Mz Model'),
    ]
    plt.legend(handles=legend_elements)
    plt.ylabel('$M_i [-]$', fontsize=32)
    plt.xlabel('$Time [ns]$', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid()
    plt.title('Transformer', fontsize=48)
    plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\same time\\problem {} curve.png'.format(sample_idx + 1), format='png', bbox_inches='tight')

    plt.plot(timeline, np.mean(sample[:,0].reshape(sample.shape[0],-1), axis=1), 'r')
    plt.plot(timeline, np.mean(sample[:,1].reshape(sample.shape[0],-1), axis=1), 'g')
    plt.plot(timeline, np.mean(sample[:,2].reshape(sample.shape[0],-1), axis=1), 'b')
    # plt.grid()
    # plt.show()
    
    plt.plot(timeline, np.mean(recon_only_out[:,0].reshape(sample.shape[0],-1), axis=1), 'rx')
    plt.plot(timeline, np.mean(recon_only_out[:,1].reshape(sample.shape[0],-1), axis=1), 'gx')
    plt.plot(timeline, np.mean(recon_only_out[:,2].reshape(sample.shape[0],-1), axis=1), 'bx')
    plt.grid()
    # plt.title('Compared to ground truth')
    legend_elements = [
        Line2D([0], [0], color='red', lw=4, label='Mx MagTense'),
        Line2D([0], [0], color='green', lw=4, label='My MagTense'),
        Line2D([0], [0], color='blue', lw=4, label='Mz MagTense'),
        Line2D([0], [0], marker='x', color='red', label='Mx Model'),
        Line2D([0], [0], marker='x', color='green', label='My Model'),
        Line2D([0], [0], marker='x', color='blue', label='Mz Model'),
    ]
    plt.legend(handles=legend_elements)
    plt.ylabel('$M_i [-]$', fontsize=32)
    plt.xlabel('$Time [ns]$', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid()
    plt.title('Auto-encoder', fontsize=48)
    plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\same time\\problem {} recon only.png'.format(sample_idx + 1), format='png', bbox_inches='tight')

    model.showMSE(sample_t, field_t, 400-init_len, True)
    model.showMSE(sample_t, field_t, 400-init_len, False)

    # time_transformer = (time_transformer_end - time_transformer_start) * 1e-9
    # print('Total time: {} s'.format(time_transformer))

if __name__ == '__main__':
    runit(0)
    runit(1)