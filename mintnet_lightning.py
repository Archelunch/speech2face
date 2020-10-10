import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.utils import make_grid

import pytorch_lightning as pl

from ranger import Ranger
import wandb


def flow_loss(u, log_jacob, size_average=True):
    log_probs = (-0.5 * u.pow(2) - 0.5 * np.log(2 * np.pi)).sum()
    log_jacob = log_jacob.sum()
    loss = -(log_probs + log_jacob)

    if size_average:
        loss /= u.size(0)
    return loss


class MintLighting(pl.LightningModule):
    def __init__(
        self,
        model,
        opt_type,
        lr,
        train_dataset,
        test_dataset,
        batch_size,
        eval_batch_size,
        n_workers,
        use_swa,
        swa_lr,
        y_condition,
        y_weight,
        warmup,
        n_init_batches,
        multi_class=False,
    ):
        super().__init__()
        self.model = model
        self.opt_type = opt_type
        self.lr = lr
        self.n_workers = n_workers
        self.eval_batch_size = eval_batch_size
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.use_swa = use_swa
        self.multi_class = multi_class
        self.swa_lr = swa_lr
        self.y_condition = y_condition
        self.y_weight = y_weight
        self.warmup = warmup
        self.n_init_batches = n_init_batches
        self.first = True

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        if self.opt_type == "Ranger":
            optimizer = Ranger(self.parameters(),
                               lr=self.lr, eps=1e-4)

        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

        return [optimizer], [scheduler]

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

    def logit_transform(self, image):
        lambd = 0.05
        image = lambd + (1 - 2 * lambd) * image
        return torch.log(image) - torch.log1p(-image)

    def sigmoid_transform(self, samples):
        lambd = 0.05
        samples = torch.sigmoid(samples)
        samples = (samples - lambd) / (1 - 2 * lambd)
        return samples

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = x * 255. / 256.
        x += torch.rand_like(x) / 256.
        x = self.logit_transform(x)

        log_det_logit = F.softplus(-x).sum() + F.softplus(x).sum() + np.prod(
            x.shape) * np.log(1 - 2 * 0.05)
        output, log_det = self.forward(x)
        loss = flow_loss(output, log_det)
        bpd = (loss.item() * x.shape[0] - log_det_logit) / \
            (np.log(2) * np.prod(x.shape)) + 8

        return {
            "loss": loss,
            "log": {"train_loss": loss, "bpd": bpd},
        }

    def sample(self):
        with torch.no_grad():
            z = torch.randn(9, 3 * 64 * 64,
                            device=self.device)
            samples = self.model.sampling(z)
            samples = self.sigmoid_transform(samples)

        return make_grid(samples.cpu(), 3)

    def validation_step(self, batch, batch_nb):
        x, y = batch
        x = x * 255. / 256.
        x += torch.rand_like(x) / 256.
        x = self.logit_transform(x)

        log_det_logit = F.softplus(-x).sum() + F.softplus(x).sum() + np.prod(
            x.shape) * np.log(1 - 2 * 0.05)
        output, log_det = self.forward(x)
        loss = flow_loss(output, log_det)
        bpd = (loss.item() * x.shape[0] - log_det_logit) / \
            (np.log(2) * np.prod(x.shape)) + 8
        return {"val_loss": loss, "bpd": bpd}

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = torch.stack([x['val_loss']
                                for x in validation_step_outputs]).mean()
        bpds = torch.stack([x['bpd']
                            for x in validation_step_outputs]).mean()
        images = self.sample()
        print("returning images")
        return {
            'val_loss': val_loss,
            "log": {"images": [wandb.Image(images, caption="samples")],
                    "val_loss": val_loss,
                    'bpd': bpds},
        }
