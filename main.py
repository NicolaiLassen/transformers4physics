import os
from distutils.command.config import config
from msilib.schema import Error
from pathlib import Path

from omegaconf import DictConfig

import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import optim
from torch.utils.data import DataLoader, random_split

import wandb
from config.config_autoregressive import AutoregressiveConfig
from config.config_emmbeding import EmmbedingConfig
from data_utils.dataset_magnet import MicroMagnetismDataset
from data_utils.dataset_phys import PhysicalDataset
from embeddings.embedding_landau_lifshitz_gilbert import \
    LandauLifshitzGilbertEmbeddingTrainer
from embeddings.embedding_model import EmbeddingModel, EmbeddingTrainingHead
from models.transformer.phys_transformer_gpt2 import PhysformerGPT2
from models.transformer.phys_transformer_helpers import PhysformerTrain
from util.config_formater import sweep_decorate_config

# Phys trainer pipeline
# 1. Train embedding model
# 2. Load embedding model & Train transformer model

# Phys gen pipeline
# 1. load embedding model & transformer
# 2. feed transformer with past tokens and set of constants c_1, c_2 ... c_N , N = max_seq_len

EMBED_TRANING_ERROR = Error(
    "Cannot use autoregressive model when traning embed")

class PhysTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # hyper
        self.save_hyperparameters(cfg)
        self.train_embed = cfg.train_embed

        # dataset
        self.train_dataset, self.val_dataset, self.test_dataset = \
            self.configure_dataset()

        # models
        self.embedding_model = self.configure_embedding_model()

        if not self.train_embed:
            self.autoregressive_model = self.configure_autoregressive_model()

    def generate(self, past_tokens, seq_len, **kwargs):
        if self.train_embed:
            raise EMBED_TRANING_ERROR

        return self.autoregressive_model.generate(past_tokens, seq_len, kwargs)

    def forward(self, z):
        if self.train_embed:
            raise EMBED_TRANING_ERROR
        return self.autoregressive_model(z)

    def configure_dataset(self) -> PhysicalDataset:
        cfg = self.hparams

        os.path.expanduser(cfg.data_dir)

        dataset = MicroMagnetismDataset()
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        return random_split(dataset, [train_size, val_size, test_size])

    def configure_embedding_model(self) -> EmbeddingTrainingHead:
        cfg = self.hparams
        return LandauLifshitzGilbertEmbeddingTrainer(
            EmmbedingConfig(cfg.embedding)
        )

    def configure_autoregressive_model(self) -> PhysformerTrain:
        cfg = self.hparams
        return PhysformerGPT2(
            AutoregressiveConfig(cfg.autoregressive)
        )

    def configure_optimizers(self):
        cfg = self.hparams

        model_parameters = self.embedding_model.parameters() if self.train_embed \
            else self.autoregressive_model

        if cfg.opt.name == 'SGD':
            optimizer = optim.SGD(model_parameters, lr=cfg.lr.lr)
        if cfg.opt.name == 'adam':
            optimizer = optim.Adam(model_parameters, lr=cfg.lr.lr)
        if cfg.opt.name == 'adamw':
            optimizer = optim.AdamW(model_parameters, lr=cfg.lr.lr,
                                    betas=(cfg.opt.beta0,
                                           cfg.opt.beta1), eps=cfg.opt.eps,
                                    weight_decay=cfg.opt.weight_decay)
        else:
            raise NotImplementedError()

        if cfg.lr.sched is not None:
            lr_scheduler = None

            if cfg.lr.sched == 'cosine':
                lr_scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cfg.lr.epochs,
                )
            elif cfg.lr.sched == 'multistep':
                lr_scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=cfg.lr.multistep_milestones,
                )

            start_epoch = 0
            if cfg.lr.start_epoch is not None:
                start_epoch = cfg.lr.start_epoch
            if lr_scheduler is not None and start_epoch > 0:
                lr_scheduler.step(start_epoch)

            return [optimizer], [lr_scheduler_g]
        else:
            return optimizer

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

    def step(self, batch, batch_idx, mode):
        x = batch
        if self.hparams["train_embed"]:
            return self.embedding_step(x, mode)
        else:
            return self.autoregressive_step(x, mode)

    def autoregressive_step():
        raise NotImplementedError("not ready")

    def embedding_step(self, x, mode):
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
        max_epochs=cfg.lr.epochs,
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
        cfg.train_embed = True
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
    # wandb sweep sweep_embed.yaml
    # wandb sweep autoregressive.yaml
    
    LandauLifshitzGilbertEmbeddingTrainer(
        config=EmmbedingConfig()
    )
    # main()
