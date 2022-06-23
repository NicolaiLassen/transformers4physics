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
from matplotlib.pyplot import figure

from x_transformers import ContinuousTransformerWrapper, Decoder
from x_transformers import ContinuousAutoregressiveWrapper
from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding
from embedding.embedding_landau_lifshitz_gilbert_ff import LandauLifshitzGilbertEmbeddingFF
from transformer.phys_transformer_gpt2 import PhysformerGPT2

def plotModel(model, suffix, folder, test_batch_sizes = []):
    date = '00'
    time = model
    transformer_suffix = suffix
    init_len = 1

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

    def batchTest(sample, field, timeFile):
        sample_t = torch.tensor(sample).float().cuda()
        field_t = torch.zeros((sample_t.size(0),3)).float().cuda()
        field_t[:] = torch.tensor(field)

        warmup_transformer = autoregressive.generate(torch.rand((1,1,128)).cuda(),2)
        warmup_transformer = autoregressive.generate(torch.rand((1,1,128)).cuda(),25)
        warmup_embedder = model.embed(sample_t[0:1], field_t[0:1])
        warmup_embedder = model.recover(warmup_embedder)

        init = model.embed(sample_t[0:init_len], field_t[0:init_len])
        batches_times = []
        for batch_size_test in test_batch_sizes:
            test_batch = init.unsqueeze(0).repeat((batch_size_test,1,1))
            time_transformer_test_start = time_ns()
            test_huge = autoregressive.generate(test_batch,seq_len=400-init_len)
            time_transformer_test_end = time_ns()
            batches_times.append((time_transformer_test_end - time_transformer_test_start) * 1e-9)
        with open('C:\\Users\\s174270\\Documents\\plots\\auto\\{}\\{}.txt'.format(folder,timeFile), 'w') as f:
            for b,t in zip(test_batch_sizes, batches_times):
                f.write('Time for batch of {}: {} s \n'.format(b,t))
            f.close()

    def plotSample(sample, field, name, timeFile):
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
        time_transformer_start = time_ns()
        emb_seq = autoregressive.generate(init,seq_len=400-init_len)
        time_transformer_end = time_ns()
        

        time_recover_start = time_ns()
        recon = model.recover(emb_seq)
        time_recover_end = time_ns()
        recon = torch.cat([sample_t[0:init_len],recon],dim=0)
        recon = recon.detach().cpu().numpy()
        recon_x = np.mean(recon[:,0].reshape(sample.shape[0],-1), axis=1)
        crosses_zero = np.argmax(recon_x < 0)

        width = 0.002
        headwidth = 2
        headlength = 5

        figure(figsize=(16,8),dpi=140)
        plt.quiver(recon[crosses_zero,0].T, recon[crosses_zero,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0), width=width, headwidth=headwidth, headlength=headlength)
        plt.axis("scaled")
        # plt.ylabel('y', fontsize=32, rotation = 0)
        # plt.xlabel('x', fontsize=32)
        # plt.xticks(fontsize=24)
        # plt.yticks(fontsize=24)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\{}\\{} cross x0.png'.format(folder, name), format='png', bbox_inches='tight')
        
        timeline = np.arange(sample.shape[0]) * 4e-12 * 1e9

        figure(figsize=(16,9),dpi=140)
        plt.plot(timeline, np.mean(sample[:,0].reshape(sample.shape[0],-1), axis=1), 'r')
        plt.plot(timeline, np.mean(sample[:,1].reshape(sample.shape[0],-1), axis=1), 'g')
        plt.plot(timeline, np.mean(sample[:,2].reshape(sample.shape[0],-1), axis=1), 'b')
        
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
        plt.setp(plt.gca().get_legend().get_texts(), fontsize='16')
        plt.ylabel('$M_i [-]$', fontsize=32)
        plt.xlabel('$Time [ns]$', fontsize=32)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid()
        plt.title('Transformer', fontsize=48)
        plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\{}\\{} transformer curve.png'.format(folder, name), format='png', bbox_inches='tight')

        time_embed = (time_embed_end - time_embed_start) * 1e-9
        time_transformer = (time_transformer_end - time_transformer_start) * 1e-9
        time_recover = (time_recover_end - time_recover_start)* 1e-9
        with open('C:\\Users\\s174270\\Documents\\plots\\auto\\{}\\{}.txt'.format(folder,timeFile), 'w') as f:
            f.write('Time to embed: {} s \n'.format(time_embed))
            f.write('Time to generate: {} s \n'.format(time_transformer))
            f.write('Time to recover: {} s \n'.format(time_recover))
            f.write('Total time: {} s \n'.format(time_embed + time_transformer + time_recover))
            f.close()

    f = h5py.File('./problem4.h5')
    sample1 = np.array(f['0']['sequence'])
    field1 = np.array(f['0']['field'])
    sample2 = np.array(f['1']['sequence'])
    field2 = np.array(f['1']['field'])

    plotSample(sample1, field1, 'problem 1', 'problem 1 timings')
    plotSample(sample2, field2, 'problem 2', 'problem 2 timings')

    if len(test_batch_sizes) > 0:
        batchTest(sample1, field1, 'batch test')

    

def plotLosses(model, val_every_n_epoch, folder):
    path = './transformer_output/{}/{}/'.format('00',model)
    f = h5py.File(path + 'transformer_losses.h5', 'r')
    losses = np.array(f['train'])
    l = np.arange(len(losses))
    figure(figsize=(16,9),dpi=140)
    plt.plot(l,losses)
    plt.title('Training Loss', fontsize=48)
    plt.yscale('log')
    plt.ylabel('Loss', fontsize=32)
    plt.xlabel('Epoch', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid()
    plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\{}\\transformer training loss.png'.format(folder), format='png', bbox_inches='tight')

    if 'val' in f.keys():
        losses_val = np.array(f['val'])
        l = np.arange(val_every_n_epoch, (len(losses_val)+1)*val_every_n_epoch, val_every_n_epoch)
        f.close()
        figure(figsize=(16,9),dpi=140)
        plt.plot(l,losses_val)
        plt.title('Validation Loss', fontsize=48)
        plt.yscale('log')
        plt.ylabel('Loss', fontsize=32)
        plt.xlabel('Epoch', fontsize=32)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid()
        plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\{}\\transformer validation loss.png'.format(folder), format='png', bbox_inches='tight')

    f.close()

if __name__ == '__main__':
    plotLosses('no dynamics', 50, 'no dynamics')
    plotModel('no dynamics', '_500', 'no dynamics', [])
    plotLosses('5', 50, 'with dynamics')
    # plotModel('5', '_450', 'with dynamics', [4, 8, 16, 32, 64, 128, 256, 512])
    plotModel('5', '_450', 'with dynamics', [])
    plotLosses('all at once', 25, 'same time')