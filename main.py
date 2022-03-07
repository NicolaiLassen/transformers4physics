from collections import OrderedDict
import re
from telnetlib import GA
import wandb
from pathlib import Path
import hydra

from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import torch
from example_code.plot import sample_check
from models.swin_transformer import SwinTransformer
from util.magtense.prism_grid import PrismGridDataset, create_dataset

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch import nn


class Moment2Field(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.save_hyperparameters(cfg)

        # TODO Select strat
        cfg.strategy

        dataset = PrismGridDataset()
        dataset.open_hdf5(
            "\\Users\\nicol\\OneDrive\\Desktop\\master\\transformers4physics\\data\\prism_grid_dataset_224.hdf5")

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = \
            random_split(dataset, [train_size, val_size, test_size])

        self.discriminator = SwinTransformer(in_chans=4, depths=[2, 2], num_heads=[1, 2], num_classes=4*224*244)
        self.generator = SwinTransformer(in_chans=4, depths=[2, 2], num_heads=[1, 2], num_classes=1)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        cfg = self.hparams

        if cfg.opt.name == 'adamw':
            optimizer_g = optim.AdamW(self.generator.parameters(), lr=cfg.lr.lr,
                                    betas=(cfg.opt.beta0, cfg.opt.beta1), eps=cfg.opt.eps,
                                    weight_decay=cfg.opt.weight_decay)

            optimizer_d  = optim.AdamW(self.discriminator.parameters(), lr=cfg.lr.lr,
                                    betas=(cfg.opt.beta0, cfg.opt.beta1), eps=cfg.opt.eps,
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

        if optimizer_idx == 0:

            # generate images
            y_hat = self(x)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(y.size(0), 1)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(y_hat)), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})

            return output
    
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(y.size(0), 1)

            real_loss = self.adversarial_loss(self.discriminator(y), valid)

            # how well can it label as fake?
            fake = torch.zeros(y.size(0), 1)

            fake_loss = self.adversarial_loss(self.discriminator(self(x).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})

            return output
  
def train(cfg):

    model = Moment2Field(cfg)

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
            ModelCheckpoint(dirpath=checkpoint_path, monitor='loss/val', mode='min'),
        ],
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )

    trainer.fit(model)
    run.finish()


@hydra.main(config_path=".", config_name="train.yaml")
def main(cfg):
    print(cfg)
    pl.seed_everything(cfg.seed)

    net = N2H(cfg)

    dataset = PrismGridDataset()
    dataset.open_hdf5(
            "\\Users\\nicol\\OneDrive\\Desktop\\master\\transformers4physics\\data\\prism_grid_dataset_224.hdf5")

    net(dataset.x[:2])
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)

    # train(cfg)


from matplotlib import pyplot as plt
import seaborn as sns; sns.set_theme()
def showNorm(imageOut):
    ax = sns.heatmap(imageOut[3], cmap="mako")
    ax.invert_yaxis()
    plt.show()


if __name__ == '__main__':
    main()
    # create_dataset(rows=[4,7,8,14], square_grid=True, res=224)