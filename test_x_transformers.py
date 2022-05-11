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

from util.data_loader import MagDataset




if __name__ == "__main__":
    epochs = 300
    ctx = 32
    ndata_train = -1
    ndata_val = -1
    stride = 4
    train_batch_size = 512
    val_batch_size = 10
    val_every_n_epoch = 50
    save_on_val = True

    transformer_cfg = {
        'ctx': ctx,
        'emb_size': 128,
        'decoder_dim': 512,
        'depth': 10,
        'heads': 8,
        'macaron': False,
        "shift_tokens": 0,
        "ff_dropout": 0.08,
        'attn_dropout': 0.08,
    }


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

    model = ContinuousTransformerWrapper(
        dim_in=transformer_cfg["emb_size"],
        dim_out=transformer_cfg["emb_size"],
        max_seq_len=transformer_cfg["ctx"],
        attn_layers=Decoder(
            dim=transformer_cfg["decoder_dim"],
            depth=transformer_cfg["depth"], 
            heads=transformer_cfg["heads"],
            macaron=transformer_cfg["macaron"],
            shift_tokens=transformer_cfg["shift_tokens"],
            ff_dropout=transformer_cfg["ff_dropout"],
            attn_dropout=transformer_cfg["attn_dropout"],
        ),
    ).cuda()
    model = ContinuousAutoregressiveWrapper(model)

    # data
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
    embedding_model = LandauLifshitzGilbertEmbedding(EmmbedingConfig(cfg),)
    embedding_model.load_model(
        "C:\\Users\\s174270\\Documents\\transformers4physics\\outputs\\2022-05-11\\13-46-21\\ckpt\\val_6.pth"
    )
    EmmbedingConfig(cfg).to_json_file(path + "embedder_cfg.json")
    torch.save(
        embedding_model.state_dict(),
        path + "embedder.pth"
    )
    embedding_model.eval()
    embedding_model.cuda()
    for param in embedding_model.parameters():
        param.requires_grad = False

    def read_and_embbed_h5_dataset(
        file_path: str,
        embedder: EmbeddingModel,
        block_size: int,
        batch_size: int = 32,
        stride: int = 5,
        n_data: int = -1,
    ) -> torch.Tensor:
        assert os.path.isfile(file_path), "Training HDF5 file {} not found".format(
            file_path
        )

        seq = []
        fields = []
        embedded_seq = []
        with h5py.File(file_path, "r") as f:

            n_seq = 0
            for key in f.keys():
                data_series = torch.Tensor(np.array(f[key]["sequence"])).cuda()
                field = torch.Tensor(np.array(f[key]["field"][:2])).unsqueeze(0).cuda()

                with torch.no_grad():
                    embedded_series = embedder.embed(data_series, field)

                # Truncate in block of block_size
                for i in range(0, data_series.size(0) - block_size + 1, stride):
                    seq.append(data_series[i : i + block_size].unsqueeze(0))
                    fields.append(field)
                    embedded_seq.append(embedded_series[i : i + block_size].unsqueeze(0))

                n_seq = n_seq + 1
                if (
                    n_data > 0 and n_seq >= n_data
                ):  # If we have enough time-series samples break loop
                    break

        seq_tensor = torch.cat(seq, dim=0).cpu()
        fields_tensor = torch.cat(fields, dim=0).cpu()
        embedded_tensor = torch.cat(embedded_seq, dim=0).cpu()
        data = MagDataset(seq_tensor, fields_tensor, embedded_tensor)

        if seq_tensor.size(0) < batch_size:
            batch_size = seq_tensor.size(0)

        return data, batch_size

    train_set, train_batch_size = read_and_embbed_h5_dataset(
        "C:\\Users\\s174270\\Documents\\datasets\\64x16 field\\field_s_state_train_circ.h5",
        embedding_model,
        block_size=ctx,
        batch_size=train_batch_size,
        stride=stride,
        n_data=ndata_train,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
        num_workers=1,
    )
    val_set, val_batch_size = read_and_embbed_h5_dataset(
        "C:\\Users\\s174270\\Documents\\datasets\\64x16 field\\field_s_state_test_circ.h5",
        embedding_model,
        block_size=ctx,
        batch_size=val_batch_size,
        stride=ctx,
        n_data=ndata_val,
    )
    val_loader = DataLoader(
        val_set,
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
    for epoch in range(epochs):
        acc_loss = 0
        bar.widgets[0] = 'Training   Epoch {:4d}/{:4d}'.format(epoch+1, epochs)
        # bar.start()
        for i, x in enumerate(train_loader):
            e = x["embedded"].cuda()
            optimizer.zero_grad()
            loss = model(e)
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
            model.eval()
            bar.widgets[0] = 'Validation Epoch {:4d}/{:4d}'.format(epoch+1, epochs)
            # bar.start()
            for i_val, x_val in enumerate(val_loader):
                e = x_val['embedded'].cuda()
                loss_val = model(e)
                acc_loss_val = acc_loss_val + loss_val.item()
                bar.widgets[-1] = '{:8f}'.format(loss_val.item())
            # bar.finish()
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