from datetime import datetime
import json
import os
import torch
import progressbar
import torch.functional as F
from torch import optim
from torch.utils.data import DataLoader
import h5py
import numpy as np
from x_transformers import ContinuousTransformerWrapper, Decoder
from x_transformers import ContinuousAutoregressiveWrapper
from config.config_emmbeding import EmmbedingConfig
from embedding.embedding_landau_lifshitz_gilbert import LandauLifshitzGilbertEmbedding
from embedding.embedding_model import EmbeddingModel
from transformer.all_in_one import AllInOne

from util.data_loader import MagDataset, read_h5_dataset




if __name__ == "__main__":
    epochs = 300
    ctx = 16
    ndata_train = 500
    ndata_val = 50
    stride = 4
    train_batch_size = 512
    val_batch_size = 10
    val_every_n_epoch = 15
    save_on_val = True
    lambda1 = 1
    lambda2 = 1

    transformer_cfg = {
        'ctx': ctx,
        'emb_size': 32,
        'decoder_dim': 512,
        'depth': 12,
        'heads': 8,
        'macaron': False,
        "shift_tokens": 0,
        "ff_dropout": 0.08,
        'attn_dropout': 0.08,
    }

    
    cfg_json = {
        'backbone': 'ResNet',
        'backbone_dim': 256,
        'channels': 5,
        'ckpt_path': './',
        'config_name': 'embedder',
        'embedding_dim': transformer_cfg["emb_size"],
        'fc_dim': 128,
        'image_size_x': 64,
        'image_size_y': 16,
        'koopman_bandwidth': -1,
        'use_koop_net': False,
    }
    class Object(object):
        pass

    embedder_cfg = Object()
    embedder_cfg.backbone= cfg_json["backbone"]
    embedder_cfg.backbone_dim = cfg_json["backbone_dim"]
    embedder_cfg.channels= cfg_json["channels"]
    embedder_cfg.ckpt_path= cfg_json["ckpt_path"]
    embedder_cfg.config_name= cfg_json["config_name"]
    embedder_cfg.embedding_dim= cfg_json["embedding_dim"]
    embedder_cfg.fc_dim= cfg_json["fc_dim"]
    embedder_cfg.image_size_x= cfg_json["image_size_x"]
    embedder_cfg.image_size_y= cfg_json["image_size_y"]
    embedder_cfg.koopman_bandwidth= cfg_json["koopman_bandwidth"]
    embedder_cfg.use_koop_net = False if "use_koop_net" not in cfg_json else cfg_json["use_koop_net"]
    embedder_cfg.pretrained = False

    now = datetime.now()
    path = "./transformer_output/{}/{}/".format(
        now.strftime("%Y-%m-%d"), now.strftime("%H-%M-%S")
    )
    os.makedirs(
        path,
        exist_ok=True,
    )
    with open(path + 'transformer_config.json', 'w') as file:
        json_data = json.dumps(transformer_cfg)
        json.dump(json_data, file)
    EmmbedingConfig(embedder_cfg).to_json_file(path + "embedder_cfg.json")

    model = AllInOne(128,transformer_cfg,embedder_cfg).cuda()

    # data
    dataset_train = 'C:\\Users\\s174270\\Documents\\datasets\\64x16 field\\field_s_state_train_circ_paper.h5'
    with open(path + 'dataset.txt','w') as file:
        file.write(dataset_train)
        file.close()

    train_dataset = read_h5_dataset(
        dataset_train,
        block_size=ctx,
        batch_size=train_batch_size,
        stride=stride,
        n_data=ndata_train,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
        num_workers=1,
    )

    mu = torch.tensor(
        [
            torch.mean(train_dataset[:]["states"][:, :, 0]),
            torch.mean(train_dataset[:]["states"][:, :, 1]),
            torch.mean(train_dataset[:]["states"][:, :, 2]),
            torch.mean(train_dataset[:]["fields"][:, 0]),
            torch.mean(train_dataset[:]["fields"][:, 1]),
            # torch.mean(self.train_dataset[:]["fields"][:, 2]),
        ]
    )
    std = torch.tensor(
        [
            torch.std(train_dataset[:]["states"][:, :, 0]),
            torch.std(train_dataset[:]["states"][:, :, 1]),
            torch.std(train_dataset[:]["states"][:, :, 2]),
            torch.std(train_dataset[:]["fields"][:, 0]),
            torch.std(train_dataset[:]["fields"][:, 1]),
            # torch.std(self.train_dataset[:]["fields"][:, 2]),
        ]
    )
    model.autoencoder.mu = mu.cuda()
    model.autoencoder.std = std.cuda()
    val_dataset = read_h5_dataset(
        "C:\\Users\\s174270\\Documents\\datasets\\64x16 field\\field_s_state_test_circ_paper.h5",
        block_size=ctx,
        batch_size=val_batch_size,
        stride=ctx,
        n_data=ndata_val,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
        num_workers=1,
    )

    # Training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(
    #     optimizer, gamma=0.995
    # )


    train_losses = []
    val_losses = []

    l_train_loader = len(train_loader)
    l_val_loader = len(val_loader)
    
    terminal_width = os.get_terminal_size().columns
    bar = progressbar.ProgressBar(
        maxval=l_train_loader,
        widgets=['Training   Epoch    1/{:4d}'.format(epochs), '   Step: ', progressbar.Counter('%3d'), ' / {:4d}    '.format(l_train_loader), progressbar.Bar('=','[',']'), ' ', progressbar.Percentage(), '    ', progressbar.Timer(), '    Loss: ', '99999999'],
        term_width=terminal_width,
    )
    model.train()
    bar.start()
    mse = torch.nn.MSELoss()
    for epoch in range(epochs):
        acc_loss = 0
        bar.widgets[0] = 'Training   Epoch {:4d}/{:4d}'.format(epoch+1, epochs)
        # bar.start()
        for i, x in enumerate(train_loader):
            optimizer.zero_grad()
            seq = x["states"].cuda()
            f = x["fields"].cuda()
            xi, xi_r, xo, xo_r, xi_h, xo_h = model(seq, f)
            loss_rec1 = mse(xi_r, xi)
            loss_rec2 = mse(xo_r, xo)
            loss_hspace = mse(xi_h, xo_h)
            loss = lambda1 * 0.5 * (loss_rec1 + loss_rec2) + lambda2 * loss_hspace
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            # lr_scheduler.step()
            acc_loss = acc_loss + loss.item()
            bar.widgets[-1] = '{:8f}'.format(loss.item())
            bar.update(i+1)
        # bar.finish()
        train_losses.append(acc_loss/l_train_loader)
        acc_loss_val = 0
        if epoch % val_every_n_epoch == val_every_n_epoch - 1:
            with torch.no_grad():
                bar.widgets[0] = 'Validation Epoch {:4d}/{:4d}'.format(epoch+1, epochs)
                for i_val, x_val in enumerate(val_loader):
                    seq = x_val["states"].cuda()
                    f = x_val["fields"].cuda()
                    xi, xi_r, xo, xo_r, xi_h, xo_h = model(seq, f)
                    loss_rec1 = mse(xi_r, xi)
                    loss_rec2 = mse(xo_r, xo)
                    loss_hspace = mse(xi_h, xo_h)
                    loss_val = lambda1 * 0.5 * (loss_rec1 + loss_rec2) + lambda2 * loss_hspace
                    acc_loss_val = acc_loss_val + loss_val.item()
                    bar.widgets[-1] = '{:8f}'.format(loss_val.item())
                model.train()
                val_losses.append(acc_loss_val/l_val_loader)
                if save_on_val:
                    torch.save(
                        model.state_dict(),
                        path + "transformer_{}.pth".format(epoch + 1),
                    )
                    torch.save(
                        optimizer.state_dict(),
                        path + "optimizer_{}.pth".format(epoch + 1),
                    )
                    f = h5py.File(
                        path + "transformer_losses.h5",
                        "w",
                    )
                    f.create_dataset("train", data=np.array(train_losses))
                    f.create_dataset("val", data=np.array(val_losses))
                    f.close()
    bar.finish()
    torch.save(
        model.state_dict(),
        path + "transformer.pth",
    )
    f = h5py.File(
        path + "transformer_losses.h5",
        "w",
    )
    f.create_dataset("train", data=np.array(train_losses))
    f.create_dataset("val", data=np.array(val_losses))
    f.close()
    print('done')
    print('Loss train: {}'.format(train_losses[-1]))
    print('Loss val  : {}'.format(val_losses[-1]))