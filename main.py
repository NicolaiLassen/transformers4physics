import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from data_utils.dataset_magnet import MicroMagnetismDataset
from data_utils.dataset_phys import PhysicalDataset
from embeddings.embedding_landau_lifshitz_gilbert import \
    LandauLifshitzGilbertEmbeddingTrainer
from embeddings.embedding_model import EmbeddingTrainingHead
from models.transformer.attention import Tensor
from models.transformer.phys_transformer_gpt2 import PhysformerGPT2
from models.transformer.phys_transformer_helpers import PhysformerTrain
from tests.koopman_git_2.utils.trainer import Trainer

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
        self.embedding_model = self.configure_embedding_model()
        # self.autoregressive_model = self.configure_autoregressive_model()

    def generate(self, past_tokens, seq_len, **kwargs):
        return self.autoregressive_model.generate(past_tokens, seq_len, kwargs)

    def forward(self, z):
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
            config=cfg.model,
        )

    def configure_autoregressive_model(self) -> PhysformerTrain:
        cfg = self.hparams
        return PhysformerGPT2()

    def configure_optimizers(self):
        cfg = self.hparams

        if cfg.opt.name == 'adamw':
            optimizer = optim.AdamW(self.generator.parameters(), lr=cfg.lr.lr,
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
                    milestones=cfg.lr.multistep_milestones
                )

            start_epoch = 0
            if cfg.lr.start_epoch is not None:
                start_epoch = cfg.lr.start_epoch
            if lr_scheduler is not None and start_epoch > 0:
                lr_scheduler.step(start_epoch)

            return optimizer, lr_scheduler_g
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
        self.embed_step(x)

    def embed_step(self, x):
        return self.embedding_model(x)


def train(cfg):

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


@hydra.main(config_path=".", config_name="train.yaml")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)


if __name__ == '__main__':
    # sweep_id = ""
    # wandb.agent(sweep_id, main, count=5, project="", entity="transformers4physics")
    main()
