
import torch
import wandb
from pathlib import Path
import hydra

from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from models.swin_unet_transformer import SwinUnetTransformer
from util.magtense.prism_grid import create_dataset, PrismGridDataset

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


# moment -> vec embed -> field
class N2H(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.net = self.get_model()

        dataset = PrismGridDataset(
            cfg.dataset, cfg.field_input, cfg.action_input, cfg.target)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.05 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = \
            random_split(dataset, [train_size, val_size, test_size])

        self.net = self.get_model()
        self.criterion = self.get_criterion()

    def get_model(self):
        # CONFIG
        return SwinUnetTransformer(in_chans=4)

    def get_criterion(self):
        return F.mse_loss

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        cfg = self.hparams

        if cfg.opt.name == 'adamw':
            optimizer = optim.AdamW(self.net.parameters(), lr=cfg.lr.lr,
                                    betas=(cfg.opt.beta0,
                                           cfg.opt.beta1), eps=cfg.opt.eps,
                                    weight_decay=cfg.opt.weight_decay)
        else:
            raise NotImplementedError()

        # setup learning rate schedule and starting epoch
        if cfg.lr.sched is not None:
            lr_scheduler = None
            if cfg.lr.sched == 'cosine':
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
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
            return [optimizer], [lr_scheduler]
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

    def training_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode='test')

    def step(self, batch, batch_idx, mode):
        x, y = batch
        x_hat = self(x)
        loss = self.criterion(x_hat, y)
        return loss

def train(cfg):
    model = N2H(cfg)
    run = wandb.init(
        project='static-moment_2_field',
        entity='transformers4physics',
        name=cfg.experiment,
        notes='test-run',
        config=cfg)
    logger = WandbLogger(log_model=True)
    logger.watch(model)
    wandb.config.update(cfg)

    checkpoint_path = Path(cfg.checkpoint_path)
    trainer = pl.Trainer(
        max_epochs=cfg.lr.epochs,
        gpus=1,
        num_nodes=1,
        logger=logger if cfg.use_wandb else None,
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
    print(cfg)
    pl.seed_everything(cfg.seed)
    train(cfg)


if __name__ == '__main__':
    from torch.nn import functional as F
    import seaborn as sns
    import matplotlib.pyplot as plt
    a = create_dataset(set_size=2, columns=[2], rows=[2], res=224)
    print(a.images_in)

    net = SwinUnetTransformer(in_chans=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    losses = []
    epochs = 100
    for i in range(epochs):
        e1 = net(a.images_in)
        l = F.mse_loss(e1, a.images_target)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(l.item())
        print(l)

    sns.lineplot(x=range(epochs), y=losses)
    plt.title('Overfit n to h test')
    plt.xlabel('epoch')
    plt.ylabel('mse loss')
    plt.show()
