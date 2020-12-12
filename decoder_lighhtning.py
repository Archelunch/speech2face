import random

import torch
import torch.optim as optim

import torch.utils.data as data
from torchvision.utils import make_grid
import pytorch_lightning as pl

import wandb
from ranger import Ranger
from datasets import postprocess

class DecoderLighting(pl.LightningModule):
    def __init__(
        self,
        glow,
        res_to_glow,
        lr,
        train_dataset,
        test_dataset,
        batch_size,
        eval_batch_size,
        n_workers,
    ):
        super().__init__()
        self.res_to_glow = res_to_glow
        self.glow = glow
        self.lr = lr
        self.n_workers = n_workers
        self.eval_batch_size = eval_batch_size
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.train_dataset = train_dataset

    def forward(self, x):
        return self.res_to_glow(x)

    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(), lr=self.lr)

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
        sound_emb, image = batch  # x embedding, y image
        z_sound = self.res_to_glow(sound_emb)  # from resemblyzer to z space
        z_image,_,_ = self.glow(x=image, reverse=False)  # from image to z space
        #print('Z image ',z_image.shape, 'Z sound ',z_sound.shape, 'SHAPE IMG',image.shape )
        image_from_sound = self.glow(z=z_sound, y_onehot=None, temperature=1, reverse=True)  # from z space (resemblyzer) to image

        image_from_sound = postprocess(image_from_sound)
        #print('SHAPE',z_image.shape, z_sound.shape)
        loss1 = torch.nn.KLDivLoss()(z_image, z_sound)
        loss2 = torch.nn.MSELoss()(image_from_sound, image)

        
        return {
            "loss": loss1 + loss2,
            "log": {"MSE_loss": loss2, "Distance Loss": loss1},
        }

    def validation_step(self, batch, batch_nb):
        sound_emb, image = batch  # x embedding, y image
        z_sound = self.res_to_glow(sound_emb)  # from resemblyzer to z space
        z_image,_,_ = self.glow(x=image, reverse=False)  # from image to z space
        #print('Z image ',z_image.shape, 'Z sound ',z_sound.shape, 'SHAPE IMG',image.shape )
        image_from_sound = self.glow(z=z_sound, y_onehot=None, temperature=1, reverse=True)  # from z space (resemblyzer) to image

        image_from_sound = postprocess(image_from_sound)
        #print('SHAPE',z_image.shape, z_sound.shape)
        loss1 = torch.nn.KLDivLoss()(z_image, z_sound)
        loss2 = torch.nn.MSELoss()(image_from_sound, image)

        return {"val_loss": loss1 + loss2}
