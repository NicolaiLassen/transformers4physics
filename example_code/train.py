import hydra
import wandb
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from poolformer import poolformer_s12
from resnet3D import ResNet, BasicBlock
from dataset import HalbachCylinderDataset


class HalbachCylinderModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.net = self.get_model()

        dataset = HalbachCylinderDataset(cfg.dataset, cfg.field_input, cfg.action_input, cfg.target)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.05 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = \
            random_split(dataset, [train_size, val_size, test_size])
    
    def get_model(self):
        if self.hparams.model.name == 'poolformer_s12':
            model = poolformer_s12(
                pretrained=self.hparams.model.pretrained,
                num_classes=self.hparams.model.num_classes,
                drop_rate=self.hparams.model.drop,
                drop_path_rate=self.hparams.model.drop_path,
                fork_feat=self.hparams.model.fork_feat,
                use_layer_scale=self.hparams.model.use_layer_scale,
                token_method=self.hparams.model.token_method,
                field_input=self.hparams.field_input,
                action_input=self.hparams.action_input,
                pos_embedding=self.hparams.pos_embedding,
                in_patch_size=self.hparams.model.in_patch_size,
                in_stride=self.hparams.model.in_stride,
                in_pad=self.hparams.model.in_pad,
            )
        elif self.hparams.model.name == 'resnet10':
            model = ResNet(
                block=BasicBlock,
                layers=[1, 1, 1, 1],
                block_inplanes=[16, 32, 64, 128],
                no_max_pool=self.hparams.model.no_max_pool,
                n_classes=self.hparams.model.num_classes,
                field_input=self.hparams.field_input,
                action_input=self.hparams.action_input,
            )
        else:
            raise NotImplementedError()
        return model

    def configure_optimizers(self):
        cfg = self.hparams

        if cfg.opt.name == 'adamw':
            optimizer = optim.AdamW(self.net.parameters(), lr=cfg.lr.lr, 
                betas=(cfg.opt.beta0, cfg.opt.beta1), eps=cfg.opt.eps,
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

    def forward(self, field, action):
        return self.net(field, action)

    def training_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode='train')
        
    def validation_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx, mode='test')

    def step(self, batch, batch_idx, mode):
        input, action, target = batch
        output = self(input, action)
        if self.hparams.target == 'p2p_next':
            loss = F.mse_loss(output, target)
            err = torch.mean(torch.abs(output - target) / target) * 100
            self.log_dict({
                f'loss/{mode}': loss.item(),
                f'err/{mode}': err.item(),
            })
        elif self.hparams.target == 'bins':
            loss = F.cross_entropy(output, target)
            _, pred = output.topk(1, 1, largest=True, sorted=True)
            correct = pred.t().eq(target.view(1, -1))
            acc = correct.float().sum() / target.size(0)
            self.log_dict({
                f'loss/{mode}': loss.item(),
                f'acc/{mode}': acc.item(),
            })
        
        return loss


def train(cfg):
    model = HalbachCylinderModel(cfg)
    run = wandb.init(
        project='SV-Halbach-p2p-next',
        entity='rl4halbach',
        name=cfg.experiment,
        notes='test-run', config=cfg)
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
            ModelCheckpoint(dirpath=checkpoint_path, monitor='loss/val', mode='min'),
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