import os
from pathlib import Path
from typing import Tuple

import h5py
import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torch.utils.data import DataLoader

from config.config_autoregressive import AutoregressiveConfig
from config.config_emmbeding import EmmbedingConfig
from embeddings.embedding_landau_lifshitz_gilbert import \
    LandauLifshitzGilbertEmbeddingTrainer
from embeddings.embedding_model import EmbeddingTrainingHead
from models.transformer.phys_transformer_gpt2 import PhysformerGPT2
from models.transformer.phys_transformer_helpers import PhysformerTrain
from util.config_formater import sweep_decorate_config


class PhysData():
    def __init__(self, data, mu, std):
        self.data = data
        self.mu = mu
        self.std = std


class PhysTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # hyper
        self.save_hyperparameters(cfg)
        self.train_embedding = cfg.train_embedding

        print(cfg)

        # dataset
        self.train_dataset, self.val_dataset, self.test_dataset = \
            self.configure_dataset()

        # models
        self.embedding_model = self.configure_embedding_model()

        if not self.train_embedding:
            self.embedding_model.eval()
            self.autoregressive_model = self.configure_autoregressive_model()

    def generate(self, past_tokens, seq_len, **kwargs):
        assert self.train_embedding, "Cannot use autoregressive model when traning embed"
        return self.autoregressive_model.generate(past_tokens, seq_len, kwargs)

    def forward(self, z):
        assert self.train_embedding, "Cannot use autoregressive model when traning embed"
        return self.autoregressive_model(z)

    def configure_dataset(self) -> Tuple[PhysData, PhysData, PhysData]:
        cfg = self.hparams

        base_path = "C:\\Users\\s174270\\Documents\\datasets\\32x32 with field"
        train_path = "{}\\train.h5".format(base_path)
        val_path = "{}\\val.h5".format(base_path)
        test_path = "{}\\test.h5".format(base_path)

        train_set = self.read_dataset(train_path,
            cfg.learning.block_size_train, 
            cfg.learning.stride_train,
            cfg.learning.batch_size_train
            )
        val_set = self.read_dataset(val_path,
            cfg.learning.block_size_val, 
            cfg.learning.stride_val,
            cfg.learning.batch_size_val
        )
        test_set = self.read_dataset(test_path,
            cfg.learning.block_size_val, 
            cfg.learning.stride_val,
            cfg.learning.batch_size_val
        )
        return train_set, val_set, test_set

    def read_dataset(self,
                     file_path: str,
                     block_size: int,
                     batch_size: int = 32,
                     stride: int = 1) -> PhysData:
        assert os.path.isfile(
            file_path), "Training HDF5 file {} not found".format(file_path)

        seq = []
        with h5py.File(file_path, "r") as f:
            for key in f.keys():
                data_series = torch.Tensor(f[key])
                # Truncate in block of block_size
                for i in range(0,  data_series.size(0) - block_size + 1, stride):
                    seq.append(data_series[i: i + block_size].unsqueeze(0))

        data = torch.cat(seq, dim=0)
        mu = torch.tensor([torch.mean(data[:, :, 0]), torch.mean(
            data[:, :, 1]), torch.mean(data[:, :, 2])])
        std = torch.tensor([torch.std(data[:, :, 0]), torch.std(
            data[:, :, 1]), torch.std(data[:, :, 2])])

        if data.size(0) < batch_size:
            print("log")
            batch_size = data.size(0)

        return PhysData(data, mu, std)

    def configure_embedding_model(self) -> EmbeddingTrainingHead:
        cfg = self.hparams
        return LandauLifshitzGilbertEmbeddingTrainer(EmmbedingConfig(cfg.embedding))

    def configure_autoregressive_model(self) -> PhysformerTrain:
        cfg = self.hparams
        return PhysformerGPT2(AutoregressiveConfig(cfg.autoregressive))

    def configure_optimizers(self):
        cfg = self.hparams

        model_parameters = self.embedding_model.parameters() if self.train_embedding \
            else self.autoregressive_model.parameters()

        if cfg.opt.name == 'SGD':
            optimizer = optim.SGD(
                model_parameters, momentum=cfg.opt.momentum, lr=cfg.learning.lr)
        if cfg.opt.name == 'adam':
            optimizer = optim.Adam(model_parameters, lr=cfg.learning.lr)
        if cfg.opt.name == 'adamw':
            optimizer = optim.AdamW(model_parameters, lr=cfg.learning.lr,
                                    betas=(cfg.opt.beta0,
                                           cfg.opt.beta1), eps=cfg.opt.eps,
                                    weight_decay=cfg.opt.weight_decay)
        else:
            raise NotImplementedError()

        if cfg.learning.sched is not None:
            lr_scheduler = None

            if cfg.learning.sched == 'exponential':
                lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=cfg.learning.gamma
                )
            elif cfg.learning.sched == 'cosine':
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    eta_min=cfg.learning.min_lr
                )
            elif cfg.learning.sched == 'multistep':
                lr_scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=cfg.learning.multistep_milestones,
                    gamma=cfg.learning.gamma
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
        return DataLoader(
            self.train_dataset.data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.hparams.pin_mem,
            num_workers=self.hparams.workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset.data,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=self.hparams.pin_mem,
            num_workers=self.hparams.workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset.data,
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

    def step(self, batch, batch_idx, mode):
        cfg = self.hparams
        x = batch

        if cfg.train_embbeding:
            return self.embedding_step(x, mode)
        else:
            return self.autoregressive_step(x, mode)

    def autoregressive_step():
        raise NotImplementedError("not ready")

    def embedding_step(self, x, mode):

        if mode == "val":
            loss = self.embedding_model.evaluate(x)
        else:
            loss, loss_reconstruct = self.embedding_model(x)

        self.log_dict({
            f'loss/{mode}': loss.item()
        })

        return loss


def train(cfg):
    pl.seed_everything(cfg.seed)
    model = PhysTrainer(cfg)

    logger = None
    if cfg.use_wandb:
        run = wandb.init(
            name=cfg.experiment,
            project=cfg.project,
            entity='transformers4physics',
            notes=cfg.notes,
            config=cfg
        )
        logger = WandbLogger(log_model=True)
        logger.watch(model)
        wandb.config.update(cfg)

    trainer = pl.Trainer(
        accumulate_grad_batches=1,
        gradient_clip_val=0.1,
        max_epochs=cfg.learning.epochs,
        gpus=1,
        num_nodes=1,
        logger=logger,
        callbacks=[
            ModelCheckpoint(dirpath=Path(cfg.checkpoint_path),
                            monitor='loss/val', mode='min')
        ],
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )

    trainer.fit(model)

    if cfg.use_wandb:
        run.finish()


def sweep_autoregressive(cfg: DictConfig):
    # wandb sweep autoregressive.yaml
    sweep = None
    with wandb.init(config=sweep):
        sweep = wandb.config
        cfg = sweep_decorate_config(cfg, sweep)
        train(cfg)

def sweep_embedding(cfg: DictConfig):
    # wandb sweep sweep_embed.yaml
    sweep = None
    with wandb.init(config=sweep):
        sweep = wandb.config
        cfg.train_embedding = True
        cfg = sweep_decorate_config(cfg, sweep)
        train(cfg)


@hydra.main(config_path=".", config_name="train.yaml")
def main(cfg: DictConfig):
    if cfg.use_sweep:
        # sweep_embedding
        wandb.agent("", sweep_embedding, count=100,
                    project="v1", entity="transformers4physics")

        # sweep_autoregressive
        # TODO
    else:
        train(cfg)

if __name__ == '__main__':
    main()
