
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from CoAtNet import CoAtEncoder, CoAtDecoder

# moment -> vec embed -> field


class N2H(pl.LightningModule):
	def __init__(self, cfg):
		super().__init__()

		self.net = self.get_model()

	def get_model(self):
		## TODO MAKE VOLUME IF config
		# CoAtEncoderDecoder
		# N -> E -> K -> D -> H
		# Koopman eigenfunctions (TODO)
		net = nn.Sequential(
				CoAtEncoder(),
				CoAtDecoder()
			)
		return net

	def forward(self, x):
		return self.net(x)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, batch, batch_idx):
		return self.step(batch=batch, batch_idx=batch_idx, mode='train')
		
	def validation_step(self, batch, batch_idx):
		return self.step(batch=batch, batch_idx=batch_idx, mode='val')

	def test_step(self, batch, batch_idx):
		return self.step(batch=batch, batch_idx=batch_idx, mode='test')

	def step(self, batch, batch_idx, mode):
		x, y = batch
		x_hat = self(x)
		loss = F.mse_loss(x_hat, y)
		return loss
