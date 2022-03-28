from pathlib import Path
from tabnanny import verbose
from typing import Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torch.utils.data import DataLoader

from config.config_autoregressive import AutoregressiveConfig
from config.config_emmbeding import EmmbedingConfig
from embedding.embedding_landau_lifshitz_gilbert import (
<<<<<<< HEAD:main.py
    LandauLifshitzGilbertEmbedding,
    LandauLifshitzGilbertEmbeddingTrainer,
)
from embedding.embedding_model import EmbeddingTrainingHead
from transformer.phys_transformer import PhysformerTrain
=======
    LandauLifshitzGilbertEmbedding, LandauLifshitzGilbertEmbeddingTrainer)
from embedding.embedding_model import EmbeddingModel, EmbeddingTrainingHead
from transformer.phys_transformer import Physformer, PhysformerTrain
>>>>>>> f1572a451ff980c7ff72a56e905d92f6a1158b32:autoregressive_trainer.py
from transformer.phys_transformer_gpt2 import PhysformerGPT2
from util.config_formater import sweep_decorate_config
from util.data_loader import PhysData, read_h5_dataset
from viz.viz_magnet import MicroMagViz

Tensor = torch.Tensor


class AutoRegressivePhysTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # hyper
        self.save_hyperparameters(cfg)

        self.lr = cfg.learning.lr
        self.batch_size = cfg.learning.batch_size_train

        # dataset
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = self.configure_dataset()

        # viz
        self.viz = MicroMagViz(cfg.viz_dir)

        # models
        self.embedding_model = self.configure_embedding_model()
        self.embedding_model.mu = self.train_dataset.mu
        self.embedding_model.std = self.train_dataset.std
        self.embedding_model_trainer = LandauLifshitzGilbertEmbeddingTrainer(
            self.embedding_model
        )

        self.autoregressive_model = self.configure_autoregressive_model()
        self.embedding_model_trainer = PhysformerTrain(self.autoregressive_model)

    def forward(self, z: Tensor):
<<<<<<< HEAD:main.py
        assert (
            self.train_embedding
        ), "Cannot use autoregressive model when traning embed"
        return self.autoregressive_model(z)

    def generate(self, past_tokens, seq_len, **kwargs):
        assert (
            self.train_embedding
        ), "Cannot use autoregressive model when traning embed"
=======
        return self.autoregressive_model(z)

    def generate(self, past_tokens, seq_len, **kwargs):
>>>>>>> f1572a451ff980c7ff72a56e905d92f6a1158b32:autoregressive_trainer.py
        return self.autoregressive_model(past_tokens, seq_len, kwargs)

    def configure_dataset(self) -> Tuple[PhysData, PhysData, PhysData]:
        cfg = self.hparams

        base_path = "C:\\Users\\s174270\\Documents\\datasets\\32x32 with field"
        train_path = "{}\\train.h5".format(base_path)
        val_path = "{}\\test.h5".format(base_path)
        test_path = "{}\\test.h5".format(base_path)

        train_set = read_h5_dataset(
            train_path,
            cfg.learning.block_size_train,
            self.batch_size,
            cfg.learning.stride_train,
            cfg.learning.n_data_train,
        )
        val_set = read_h5_dataset(
            val_path,
            cfg.learning.block_size_val,
            self.batch_size,
            cfg.learning.stride_val,
        )
        test_set = read_h5_dataset(
            test_path,
            cfg.learning.block_size_val,
            self.batch_size,
            cfg.learning.stride_val,
        )
        return train_set, val_set, test_set

    def configure_embedding_model(self) -> EmbeddingModel:
        cfg = self.hparams
        return LandauLifshitzGilbertEmbedding(EmmbedingConfig(cfg.embedding))

    def configure_autoregressive_model(self) -> Physformer:
        cfg = self.hparams
        return PhysformerGPT2(AutoregressiveConfig(cfg.autoregressive))

    def configure_optimizers(self):
        cfg = self.hparams

<<<<<<< HEAD:main.py
        model_parameters = (
            self.embedding_model.parameters()
            if self.train_embedding
            else self.autoregressive_model.parameters()
        )
=======
        model_parameters = self.autoregressive_model.parameters()
>>>>>>> f1572a451ff980c7ff72a56e905d92f6a1158b32:autoregressive_trainer.py

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
            self.train_dataset.data,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            pin_memory=cfg.pin_mem,
            num_workers=cfg.workers,
        )

    def val_dataloader(self):
        cfg = self.hparams
        return DataLoader(
            self.val_dataset.data,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=True,
            pin_memory=cfg.pin_mem,
            num_workers=1,
        )

    def test_dataloader(self):
        cfg = self.hparams
        return DataLoader(
            self.test_dataset.data,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=True,
            pin_memory=cfg.pin_mem,
            num_workers=1,
        )

    def training_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode="test")

    def step(self, batch: Tensor, batch_idx: int, mode: str):
        x = batch

        seq = torch.zeros(x.size(0), x.size(1), 140)

        for t in range(x.size(1)):
            e = self.embedding_model.embed(x[:, t])
            seq[:, t] = e

<<<<<<< HEAD:main.py
    def embedding_step(self, x: Tensor, mode: str):
        if mode == "val":
            loss, _, _ = self.embedding_model_trainer.evaluate(x)
            self.log_dict(
                {f"embedding_loss/{mode}": loss.item()}, on_epoch=True, on_step=False,
            )
            return loss

        loss, loss_reconstruct = self.embedding_model_trainer(x)
        self.log_dict(
            {
                f"embedding_loss/{mode}": loss_reconstruct.item(),
                f"embedding_loss_koopman/{mode}": loss.item(),
            },
            on_epoch=True,
            on_step=False,
        )
        return loss
=======
        h, p = self.autoregressive_model(seq)
        print(p.shape)
        # return self.embedding_step(x, mode)
        # return self.autoregressive_step(x, mode)

    def eval_step(self, x):
        preds = torch.rand()
        self.eval_states(preds, x)

    def eval_states(self, pred_embeds: Tensor, x: Tensor ):
        bsize = pred_embeds.size(0)
        tsize = pred_embeds.size(1)

        x_in = pred_embeds.contiguous().view(-1, pred_embeds.size(-1))
        out = self.embedding_model.recover(x_in)
        out = out.view([bsize, tsize] + self.embedding_model.input_dims)
        
        mse = nn.MSELoss()
        targets_error = mse(out, x)

        ## PLOT

        return targets_error
>>>>>>> f1572a451ff980c7ff72a56e905d92f6a1158b32:autoregressive_trainer.py


def train(cfg):
    pl.seed_everything(cfg.seed)
    model = AutoRegressivePhysTrainer(cfg)

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
        precision=16,
        auto_lr_find=True,
        accumulate_grad_batches=1,
        gradient_clip_val=0.1,
        max_epochs=cfg.learning.epochs,
        gpus=cfg.gpus,
        logger=logger,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        check_val_every_n_epoch=2,
    )

    trainer.fit(model)

    if cfg.use_wandb:
        wandb.finish()


@hydra.main(config_path=".", config_name="train.yaml")
def sweep_autoregressive(cfg: DictConfig):
    # wandb sweep autoregressive.yaml
    sweep = None
    with wandb.init(config=sweep):
        sweep = wandb.config
        cfg = sweep_decorate_config(cfg, sweep)
        train(cfg)


@hydra.main(config_path=".", config_name="train.yaml")
def sweep_embedding(cfg: DictConfig):
    # wandb sweep sweep_embed.yaml
    sweep = None
    with wandb.init(config=sweep):
        sweep = wandb.config
        cfg = sweep_decorate_config(cfg, sweep)
        train(cfg)


@hydra.main(config_path=".", config_name="train.yaml")
def main(cfg: DictConfig):
    model = AutoRegressivePhysTrainer(cfg)
    model.step(torch.rand(2, 16, 3, 32, 32), 0, "train")


if __name__ == "__main__":
    # main()
    wandb.agent(
        "gv398m8m",
        sweep_embedding,
        count=50,
        project="v1",
        entity="transformers4physics",
    )

