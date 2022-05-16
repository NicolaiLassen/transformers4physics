from enum import auto
from gc import callbacks
from pathlib import Path
from tabnanny import verbose
from typing import Tuple

import hydra
import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torch.utils.data import DataLoader

from callback import SaveCallback
from config.config_emmbeding import EmmbedingConfig
from embedding.embedding_landau_lifshitz_gilbert import (
    LandauLifshitzGilbertEmbedding,
    LandauLifshitzGilbertEmbeddingTrainer,
)
from embedding.embedding_model import EmbeddingModel
from util.config_formater import sweep_decorate_config
from util.data_loader import read_h5_dataset
from viz.viz_magnet import MicroMagViz

Tensor = torch.Tensor


class EmbeddingPhysTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.viz = MicroMagViz()

        # hyper
        self.save_hyperparameters(cfg)
        self.lr = cfg.learning.lr
        self.batch_size = cfg.learning.batch_size_train
        self.train_path = ''

        # dataset
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = self.configure_dataset()

        mu = torch.tensor(
            [
                torch.mean(self.train_dataset[:]["states"][:, :, 0]),
                torch.mean(self.train_dataset[:]["states"][:, :, 1]),
                torch.mean(self.train_dataset[:]["states"][:, :, 2]),
                torch.mean(self.train_dataset[:]["fields"][:, 0]),
                torch.mean(self.train_dataset[:]["fields"][:, 1]),
                # torch.mean(self.train_dataset[:]["fields"][:, 2]),
            ]
        )
        std = torch.tensor(
            [
                torch.std(self.train_dataset[:]["states"][:, :, 0]),
                torch.std(self.train_dataset[:]["states"][:, :, 1]),
                torch.std(self.train_dataset[:]["states"][:, :, 2]),
                torch.std(self.train_dataset[:]["fields"][:, 0]),
                torch.std(self.train_dataset[:]["fields"][:, 1]),
                # torch.std(self.train_dataset[:]["fields"][:, 2]),
            ]
        )

        # model
        self.model = self.configure_embedding_model()
        self.model.mu = mu
        self.model.std = std
        self.model_trainer = LandauLifshitzGilbertEmbeddingTrainer(
            self.model,
            l1=1e1,
            l2=1e1,
            l3=1e-2,
            l4=2,
        )

        self.val_id = 0
        self.losses_train = []
        self.losses_val = []
        self.losses_train_hist = []
        self.losses_val_hist = []

    def forward(self, z: Tensor):
        return self.model.embed(z)

    def configure_dataset(self) -> Tuple[Tensor, Tensor, Tensor]:
        cfg = self.hparams

        base_path = "C:\\Users\\s174270\\Documents\\datasets\\64x16 field"
        # train_path = "{}\\field_s_state_train_large.h5".format(base_path)
        # val_path = "{}\\field_s_state_test_large.h5".format(base_path)
        # test_path = "{}\\field_s_state_test_large.h5".format(base_path)
        train_path = "{}\\field_s_state_train_circ_paper.h5".format(base_path)
        val_path = "{}\\field_s_state_test_circ_paper.h5".format(base_path)
        test_path = "{}\\field_s_state_test_circ_paper.h5".format(base_path)
        # train_path = base_path + "\\field_s_state_train_rest.h5"
        # val_path = base_path + "\\field_s_state_test_rest.h5"
        # test_path = base_path + "\\field_s_state_test_rest.h5"
        self.train_path = train_path

        train_set = read_h5_dataset(
            train_path,
            cfg.learning.block_size_train,
            self.batch_size,
            cfg.learning.stride_train,
            cfg.learning.n_data_train
        )
        val_set = read_h5_dataset(
            val_path,
            cfg.learning.block_size_val,
            self.batch_size,
            cfg.learning.stride_val,
            -1,
        )
        test_set = read_h5_dataset(
            test_path,
            cfg.learning.block_size_val,
            self.batch_size,
            cfg.learning.stride_val,
            1,
        )
        return train_set, val_set, test_set

    def configure_embedding_model(self) -> EmbeddingModel:
        cfg = self.hparams
        return LandauLifshitzGilbertEmbedding(EmmbedingConfig(cfg.embedding))

    def configure_optimizers(self):
        cfg = self.hparams

        model_parameters = self.model.parameters()

        if cfg.opt.name == "adamw":
            optimizer = optim.AdamW(
                model_parameters,
                lr=self.lr,
                betas=(cfg.opt.beta0, cfg.opt.beta1),
                eps=cfg.opt.eps,
                weight_decay=cfg.opt.weight_decay,
            )
        elif cfg.opt.name == "adam":
            optimizer = optim.Adam(model_parameters, lr=self.lr, weight_decay=1e-8)
        else:
            raise NotImplementedError()

        if cfg.learning.sched is not None:
            lr_scheduler = None

            if cfg.learning.sched == "exponential":
                lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=cfg.learning.gamma
                )
            elif cfg.learning.sched == "cosine":
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, eta_min=cfg.learning.min_lr, T_max=cfg.learning.epochs
                )

            start_epoch = 0
            if cfg.learning.start_epoch is not None:
                start_epoch = cfg.learning.start_epoch
            if lr_scheduler is not None and start_epoch > 0:
                lr_scheduler.step(start_epoch)

            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def train_dataloader(self):
        cfg = self.hparams
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            pin_memory=cfg.pin_mem,
            num_workers=cfg.workers,
        )

    def val_dataloader(self):
        cfg = self.hparams
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=True,
            pin_memory=cfg.pin_mem,
            num_workers=cfg.workers,
        )

    def test_dataloader(self):
        cfg = self.hparams
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=True,
            pin_memory=cfg.pin_mem,
            num_workers=cfg.workers,
        )

    def training_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        self.val_id = self.val_id + 1
        self.save_model(filename='val_{}'.format(self.val_id))
        return self.step(batch=batch, batch_idx=batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode="test")

    def on_epoch_end(self):
        if(len(self.losses_train) > 0):
            self.losses_train_hist.append(sum(self.losses_train)/len(self.losses_train))
        if(len(self.losses_val) > 0):
            self.losses_val_hist.append(sum(self.losses_val)/len(self.losses_val))
            f = h5py.File('./losses.h5', 'w')
            f.create_dataset('train', data=np.array(self.losses_train_hist))
            f.create_dataset('val', data=np.array(self.losses_val_hist))
            f.close()
        self.losses_train = []
        self.losses_val = []

    def step(self, batch, batch_idx: int, mode: str):
        x = batch
        s = x["states"]
        f = x["fields"]

        # if(mode=='val'):
        #     self.viz.plot_prediction(self.model(s[0],f[0].unsqueeze(0))[1],s[0])

        loss, loss_reconstruct = (
            self.model_trainer.evaluate(s, f)
            if mode == "val"
            else self.model_trainer(s, f)
        )

        self.log_dict(
            {
                f"loss_reconstruct/{mode}": loss_reconstruct.item(),
                f"loss_koopman/{mode}": loss.item(),
            },
            on_epoch=True,
            on_step=False,
        )
        if(mode == 'train'):
            self.losses_train.append(loss.item())
        elif(mode == 'val'):
            self.losses_val.append(loss_reconstruct.item())
        return loss

    def save_model(self, checkpoint_dir="./ckpt", filename="embed"):
        self.model.save_model(save_directory=checkpoint_dir, filename=filename)
        with open('dataset.txt', 'w') as file:
            file.write(str(self.train_path))
        file.close()


def train(cfg):
    pl.seed_everything(cfg.seed)
    model = EmbeddingPhysTrainer(cfg)

    logger = None
    if cfg.use_wandb:
        if wandb.run is None:
            wandb.init(
                name=cfg.experiment,
                project=cfg.project,
                entity="transformers4physics",
                notes=cfg.notes,
                config=cfg,
            )
        logger = WandbLogger(log_model=True)
        logger.watch(model)
        wandb.config.update(cfg)

    trainer = pl.Trainer(
        devices=1,
        accelerator="auto",
        accumulate_grad_batches=1,
        gradient_clip_val=0.1,
        max_epochs=cfg.learning.epochs,
        gpus=cfg.gpus,
        logger=logger,
        num_sanity_val_steps=0,
        log_every_n_steps=100,
        check_val_every_n_epoch=50,
        callbacks=SaveCallback(
            dirpath="{}".format(cfg.embedding.ckpt_path),
            filename=cfg.embedding.display_name,
        ),
    )

    trainer.fit(model)

    if cfg.use_wandb:
        wandb.finish()


@hydra.main(config_path=".", config_name="train.yaml")
def sweep(cfg: DictConfig):
    # wandb sweep sweep_embed.yaml
    sweep = None
    with wandb.init(config=sweep):
        sweep = wandb.config
        cfg.embedding.display_name = wandb.run.name
        cfg.embedding.sweep_id = wandb.run.sweep_id
        cfg = sweep_decorate_config(cfg, sweep)
        train(cfg)


@hydra.main(config_path=".", config_name="train.yaml")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
    # wandb.agent("gv398m8m", sweep, count=2, project="v1", entity="transformers4physics")
