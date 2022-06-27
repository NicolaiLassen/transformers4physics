import json
import h5py
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from config.config_emmbeding import EmmbedingConfig
from config.phys_config import PhysConfig
from matplotlib.pyplot import figure
import torch

from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding
from embedding.embedding_landau_lifshitz_gilbert_ff import LandauLifshitzGilbertEmbeddingFF
from embedding.embedding_landau_lifshitz_gilbert_ss import LandauLifshitzGilbertEmbeddingSS

def plotModel(model, model_name, folder):
        # date = '2022-05-16'
        # time = '11-19-55'
        # model_name = 'val_3'
        date = '00'
        time = model
        model_name = model_name

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

        f = h5py.File('./problem4.h5')

        with open('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\{}\\{}\\dataset.txt'.format(date,time)) as file:
            print(file.read(-1))
        
        sample1 = np.array(f['0']['sequence'])
        field1 = np.array( f['0']['field'])
        
        sample2 = np.array(f['1']['sequence'])
        field2 = np.array( f['1']['field'])
        
        def plotSample(sample,field,name):
            # print(sample.shape)
            sample_t = torch.tensor(sample).float().cuda()
            field_t = torch.zeros((sample_t.size(0),3)).float().cuda()
            field_t[:] = torch.tensor(field)
            a = model.embed(sample_t, field_t)
            recon = model.recover(a)

            normsX = torch.sqrt(torch.einsum('ij,ij->j',sample_t.swapaxes(1,3).reshape(-1,3).T, sample_t.swapaxes(1,3).reshape(-1,3).T))
            normsX = normsX.reshape(-1,16,64).swapaxes(1,2)
            sample_t[:,0,:,:] = sample_t[:,0,:,:]/normsX
            sample_t[:,1,:,:] = sample_t[:,1,:,:]/normsX
            sample_t[:,2,:,:] = sample_t[:,2,:,:]/normsX

            normsG = torch.sqrt(torch.einsum('ij,ij->j',recon.swapaxes(1,3).reshape(-1,3).T, recon.swapaxes(1,3).reshape(-1,3).T))
            normsG = normsG.reshape(-1,16,64).swapaxes(1,2)
            recon[:,0,:,:] = recon[:,0,:,:]/normsG
            recon[:,1,:,:] = recon[:,1,:,:]/normsG
            recon[:,2,:,:] = recon[:,2,:,:]/normsG

            recon = recon.detach().cpu().numpy()

            # plt.quiver(sample[0,0].T, sample[0,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
            # plt.quiver(recon[0,0].T, recon[0,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
            # plt.ylabel('y', fontsize=32, rotation = 0)
            # plt.xlabel('x', fontsize=32)
            # plt.xticks(fontsize=24)
            # plt.yticks(fontsize=24)
            # plt.show()
            # plt.quiver(sample[-1,0].T, sample[-1,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
            # plt.quiver(recon[-1,0].T, recon[-1,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
            # plt.ylabel('y', fontsize=32, rotation = 0)
            # plt.xlabel('x', fontsize=32)
            # plt.xticks(fontsize=24)
            # plt.yticks(fontsize=24)
            # plt.show()

            timeline = np.arange(sample.shape[0]) * 4e-12 * 1e9
            
            figure(figsize=(16,9),dpi=140)
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
            plt.setp(plt.gca().get_legend().get_texts(), fontsize='16')
            plt.ylabel('$M_i [-]$', fontsize=32)
            plt.xlabel('$Time [ns]$', fontsize=32)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.grid()
            plt.title('Auto-encoder', fontsize=48)
            plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\{}\\{}.png'.format(folder, name), format='png', bbox_inches='tight')
            

        plotSample(sample1, field1, 'problem 1 embedder recon')
        plotSample(sample2, field2, 'problem 2 embedder recon')

def showLosses(time, val_every_n_epoch, folder):
    date = '00'
    f = h5py.File('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\{}\\{}\\losses.h5'.format(date,time), 'r')
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
    plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\{}\\embedder training loss.png'.format(folder), format='png', bbox_inches='tight')
    losses = np.array(f['val'])
    l = np.arange(len(losses))
    l = np.arange(val_every_n_epoch, (len(losses)+1)*val_every_n_epoch, val_every_n_epoch)
    figure(figsize=(16,9),dpi=140)
    plt.plot(l,losses)
    plt.title('Validation Loss', fontsize=48)
    plt.yscale('log')
    plt.ylabel('Loss', fontsize=32)
    plt.xlabel('Epoch', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid()
    plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\{}\\embedder validation loss.png'.format(folder), format='png', bbox_inches='tight')

def plotKoop(recover_every_step = False):
    date = '00'
    time = 'circ paper static koop'
    model_name = 'val_6'

    start_at = 0
    koop_forward = 1
    f = h5py.File('./problem4.h5')
    with open('C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\{}\\{}\\dataset.txt'.format(date,time)) as file:
        print(file.read(-1))
    sample = np.array(f['0']['sequence'])
    field = np.array( f['0']['field'])
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
    if not recover_every_step:
        for i in range(1,400-start_at):
            b[i] = model.koopman_operation(b[i-1].unsqueeze(0),field_t)[0]
    else:
        for i in range(1,400-start_at):
            asd = model.koopman_operation(b[i-1].unsqueeze(0),field_t)
            asd = model.recover(asd)
            b[i] = model.embed(asd, field_t)
    # mse = torch.nn.MSELoss()
    # print(mse(model.recover(c[0]),model.recover(b[koop_forward])))
    # b[start_at+koop_forward] = c[0]
    # print(mse(sample_t, model.recover(model.embed(sample_t, field_t))))
    # print(mse(model.recover(sample_t[0]),model.recover(b[koop_to])))
    recon = model.recover(b)

    recon = recon.detach().cpu().numpy()
    b = b.detach().cpu().numpy()

    # plt.quiver(sample[start_at,0].T, sample[start_at,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    # plt.quiver(recon[start_at,0].T, recon[start_at,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    # plt.show()
    # plt.quiver(sample[koop_forward,0].T, sample[koop_forward,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    # plt.quiver(recon[koop_forward,0].T, recon[koop_forward,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    # plt.show()
    # plt.quiver(sample[-1,0].T, sample[-1,1].T, pivot='mid', color=(0.0,0.0,0.0,1.0))
    # plt.quiver(recon[-1,0].T, recon[-1,1].T, pivot='mid', color=(0.6,0.0,0.0,0.7))
    # plt.show()
    
    timeline = np.arange(start_at, sample.shape[0]) * 4e-12 * 1e9

    figure(figsize=(16,9))
    plt.plot(timeline, np.mean(sample[start_at:,0].reshape(400-start_at,-1), axis=1), 'r')
    plt.plot(timeline, np.mean(sample[start_at:,1].reshape(400-start_at,-1), axis=1), 'g')
    plt.plot(timeline, np.mean(sample[start_at:,2].reshape(400-start_at,-1), axis=1), 'b')
    # plt.grid()
    # plt.show()
    
    plt.plot(timeline, np.mean(recon[:,0].reshape(400-start_at,-1), axis=1), 'rx')
    plt.plot(timeline, np.mean(recon[:,1].reshape(400-start_at,-1), axis=1), 'gx')
    plt.plot(timeline, np.mean(recon[:,2].reshape(400-start_at,-1), axis=1), 'bx')
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
    if recover_every_step:
        plt.title('Koopman dynamics (Recover every step)', fontsize=48)
        plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\with dynamics\\instant decay reconstruction along the way.png', format='png', bbox_inches='tight')
    else:
        plt.title('Koopman dynamics (Stay in latent space)', fontsize=48)
        plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\with dynamics\\instant decay.png', format='png', bbox_inches='tight')

    plt.clf()
    figure(figsize=(16,9))
    plt.plot(np.arange(b.shape[0])[:10], np.sum(b,1)[:10])
    plt.ylabel('$\Sigma g_i$', fontsize=32)
    plt.xlabel('$i$', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid()
    if recover_every_step:
        plt.title('Sum of $g_i$ (Recover every step)', fontsize=48)
        plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\with dynamics\\sum g recon along way.png', format='png', bbox_inches='tight')
    else:
        plt.title('Sum of $g_i$ (Stay in latent space)', fontsize=48)
        plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\with dynamics\\sum g.png', format='png', bbox_inches='tight')

    plt.clf()
    figure(figsize=(16,9))
    plt.plot(np.arange(b.shape[1]), b[-1])
    plt.ylabel('$g[i]$', fontsize=32)
    plt.xlabel('$i$', fontsize=32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid()
    if recover_every_step:
        plt.title('Last g (Recover every step)', fontsize=48)
        plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\with dynamics\\last g recon along way.png', format='png', bbox_inches='tight')
    else:
        plt.title('Last g (Stay in latent space)', fontsize=48)
        plt.savefig('C:\\Users\\s174270\\Documents\\plots\\auto\\with dynamics\\last g.png', format='png', bbox_inches='tight')

    

if __name__ == '__main__':
    print('no dynamics start')
    showLosses('no dynamics', 50, 'no dynamics')
    plotModel('no dynamics', 'val_5', 'no dynamics')
    print('no dynamics done')
    print('with dynamics start')
    showLosses('circ paper static koop', 50, 'with dynamics')
    plotModel('circ paper static koop', 'val_6', 'with dynamics')
    plotKoop()
    plotKoop(recover_every_step=True)
    print('with dynamics done')