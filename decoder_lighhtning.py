import random

import torch
import torch.optim as optim

import torch.utils.data as data
from torchvision.utils import make_grid
import pytorch_lightning as pl

import wandb
from ranger import Ranger


class DecoderLighting(pl.LightningModule):
    def __init__(self, decoder, glow, lr, opt_type, train_dataset, test_dataset, batch_size, eval_batch_size, n_workers, loss_name):
        self.model = model
        self.glow = glow
        self.opt_type = opt_type
        self.lr = lr
        self.n_workers = n_workers
        self.eval_batch_size = eval_batch_size
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.loss_name = loss_name

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(),
                           lr=self.lr)

        return [optimizer]

    def train_dataloader(self):
        train_loader = data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        test_loader = data.DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            drop_last=True,
        )
        return test_loader

    def training_step(self, batch, batch_nb):
        x, y = batch  # x embedding, y image
        x = self.model(x)  # from resemblyzer to z space
        with torch.no_grad():
            y_z = self.glow(y)  # from image to z space

            # from z space (resemblyzer) to image
            z_y = self.glow(x, reverse=True)

        loss1 = torch.nn.KLDivLoss()(x, y_z)
        loss2 = torch.nn.MSE()(z_y, y)

        return {
            "loss": loss1 + loss2,
            "log": {"MSE_loss": loss2, 'Distance Loss': loss1},
        }

        def validation_step(self, batch, batch_nb):
            x, y = batch
            x = self.model(x)
            with torch.no_grad():
                y_z = self.glow(y)
                z_y = self.glow(x, reverse=True)

            loss1 = torch.nn.KLDivLoss()(x, y_z)
            loss2 = torch.nn.MSE()(z_y, y)

            return {"val_loss": loss1+loss2}
