import re
from collections import OrderedDict
from pathlib import Path
from telnetlib import GA
from typing import List

import hydra
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import R, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from data_utils.dataset_magnet import MicroMagnetismDataset
from data_utils.dataset_phys import PhysicalDataset
from embeddings.embedding_landau_lifshitz_gilbert import \
    LandauLifshitzGilbertEmbeddingTrainer
from embeddings.embedding_model import EmbeddingTrainingHead
from models.transformer.attention import Tensor
from models.transformer.phys_transformer_gpt2 import PhysformerGPT2
from models.transformer.phys_transformer_helpers import PhysformerTrain

# Phys trainer pipeline
# 1. Train embedding model
# 2. Train transformer model

# Phys gen pipeline
# 1. load embedding model & transformer
# 2. feed transformer with past tokens and set of constants c_1, c_2 ... c_N , N = max_seq_len


class PhysTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # hyper
        self.save_hyperparameters(cfg)

        # dataset
        self.train_dataset, self.val_dataset, self.test_dataset = \
            self.configure_dataset()

        # models
        self.embedding_model = self.configure_trainer_embedding_model()
        self.autoregressive_model = self.configure_trainer_autoregressive_model()

    def generate(self, past_tokens, seq_len, **kwargs):

        was_training = self.net.training
        self.autoregressive_model.eval()
        self.embedding_model.eval()

        num_dims = len(past_tokens.shape)

        if num_dims == 1:
            past_tokens = past_tokens[None, :]

        b, t = past_tokens.shape

        out = self.embedding_model.embed(past_tokens)

        # TODO
        constants_seq = kwargs.pop('constants_seq', None)
        input_mask = kwargs.pop('input_mask', None)

        if input_mask is None:
            input_mask = torch.full_like(
                out, True, dtype=torch.bool, device=out.device)

        for step in range(seq_len):

            # Notify network of constant chagnes to rebase external
            if constants_seq is not None:
                constant_embeding = self.embedding_model.recover(
                    constants_seq[step])
                # fill attribute for transformer of constants

            x = out[:, -self.max_seq_len:]
            input_mask = input_mask[:, -self.max_seq_len:]

            outputs = self.autoregressive_model(
                x, input_mask=input_mask, **kwargs)
            next_output = outputs[0][:, -1:]

            out = torch.cat((out, next_output), dim=-1)

            input_mask = F.pad(input_mask, (0, 1), value=True)

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, z):
        return self.autoregressive_model(z)

    def configure_dataset(self) -> PhysicalDataset:
        dataset = MicroMagnetismDataset()
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        return random_split(dataset, [train_size, val_size, test_size])

    def configure_trainer_embedding_model(self) -> EmbeddingTrainingHead:
        return LandauLifshitzGilbertEmbeddingTrainer()

    def configure_trainer_autoregressive_model(self) -> PhysformerTrain:
        return PhysformerGPT2()

    def configure_optimizers(self):
        cfg = self.hparams

        if cfg.opt.name == 'adamw':
            optimizer_g = optim.AdamW(self.generator.parameters(), lr=cfg.lr.lr,
                                      betas=(cfg.opt.beta0,
                                             cfg.opt.beta1), eps=cfg.opt.eps,
                                      weight_decay=cfg.opt.weight_decay)

            optimizer_d = optim.AdamW(self.discriminator.parameters(), lr=cfg.lr.lr,
                                      betas=(cfg.opt.beta0,
                                             cfg.opt.beta1), eps=cfg.opt.eps,
                                      weight_decay=cfg.opt.weight_decay)
        else:
            raise NotImplementedError()

        # setup learning rate schedule and starting epoch
        if cfg.lr.sched is not None:
            lr_scheduler = None
            if cfg.lr.sched == 'cosine':
                lr_scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_g,
                    T_max=cfg.lr.epochs,
                )
                lr_scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_d,
                    T_max=cfg.lr.epochs,
                )
            elif cfg.lr.sched == 'multistep':
                lr_scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer_g,
                    milestones=cfg.lr.multistep_milestones
                )
                lr_scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_d,
                    T_max=cfg.lr.epochs,
                )

            start_epoch = 0
            if cfg.lr.start_epoch is not None:
                start_epoch = cfg.lr.start_epoch
            if lr_scheduler is not None and start_epoch > 0:
                lr_scheduler.step(start_epoch)

            return [optimizer_g, optimizer_d], [lr_scheduler_g, lr_scheduler_d]
        else:
            return [optimizer_g, optimizer_d], []

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.hparams.pin_mem,
            num_workers=self.hparams.workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=self.hparams.pin_mem,
            num_workers=self.hparams.workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=self.hparams.pin_mem,
            num_workers=self.hparams.workers
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode='train', optimizer_idx=optimizer_idx)

    def validation_step(self, batch, batch_idx, optimizer_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode='val', optimizer_idx=optimizer_idx)

    def test_step(self, batch, batch_idx, optimizer_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode='test', optimizer_idx=optimizer_idx)

    def step(self, batch, batch_idx, mode, optimizer_idx):
        x, _, y = batch

        self.embedding_model.train()
        self.autoregressive_model.train()


def train(cfg):

    model = PhysTrainer(cfg)

    logger = None
    if cfg.use_wandb:
        run = wandb.init(
            name=cfg.experiment,
            project=cfg.project,
            entity='transformers4physics',
            notes=cfg.notes,
            config=cfg)
        logger = WandbLogger(log_model=True)
        logger.watch(model)
        wandb.config.update(cfg)

    checkpoint_path = Path(cfg.checkpoint_path)

    trainer = pl.Trainer(
        max_epochs=cfg.lr.epochs,
        gpus=1,
        num_nodes=1,
        logger=logger,
        callbacks=[
            ModelCheckpoint(dirpath=checkpoint_path,
                            monitor='loss/val', mode='min'),
        ],
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )

    trainer.fit(model)
    run.finish()


@hydra.main(config_path=".", config_name="train.yaml")
def main(cfg):
    pl.seed_everything(cfg.seed)
    train(cfg)


if __name__ == '__main__':
    main()
