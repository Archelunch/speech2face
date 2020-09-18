import torch
import torch.optim as optim
import torch.utils.data as data
from torchvision.utils import make_grid

import pytorch_lightning as pl

import wandb

def compute_loss(nll, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    if multi_class:
        y_logits = torch.sigmoid(y_logits)
        loss_classes = F.binary_cross_entropy_with_logits(
            y_logits, y, reduction=reduction
        )
    else:
        loss_classes = F.cross_entropy(
            y_logits, torch.argmax(y, dim=1), reduction=reduction
        )

    losses["loss_classes"] = loss_classes
    losses["total_loss"] = losses["nll"] + y_weight * loss_classes

    return losses

class GlowLighting(pl.LightningModule):
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

    def forward(self, x=None, y_onehot=None, z=None, temperature=None, reverse=False):
        return self.model.forward(
            x=x, y_onehot=y_onehot, z=z, temperature=temperature, reverse=reverse
        )

    def configure_optimizers(self):
        """TODO SWA"""
        if self.opt_type == "AdamW":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(epoch):
            return min(1.0, (epoch + 1) / self.warmup)  # noqa

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        # if self.use_swa:
        # swa_model = AveragedModel(self.model)
        # swa_scheduler = SWALR(optimizer, swa_lr=self.swa_lr

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

    def training_step(self, batch, batch_nb):
        x, y = batch
        if self.y_condition:
            z, nll, y_logits = self.forward(x, y)
            losses = compute_loss_y(nll, y_logits, self.y_weight, y, self.multi_class)
        else:
            z, nll, y_logits = self.forward(x, None)
            losses = compute_loss(nll)

        return {
            "loss": losses["total_loss"],
            "log": {"train_loss": losses["total_loss"]},
        }

    def sample(self):
        with torch.no_grad():
            if self.y_condition:
                y = torch.eye(num_classes)
                y = y.repeat(self.batch_size // num_classes + 1)
                y = y[:32, :].to(device)  # number hardcoded in model for now
            else:
                y = None

            images = self.model(y_onehot=y, temperature=1, reverse=True)
        return (
            make_grid(images.cpu()[:30], nrow=6, normalize=False)
            .permute(1, 2, 0)
            .numpy()
        )

    def validation_step(self, batch, batch_nb):
        x, y = batch
        if self.y_condition:
            z, nll, y_logits = self.forward(x, y)
            losses = compute_loss_y(
                nll, y_logits, self.y_weight, y, self.multi_class, reduction="none"
            )
        else:
            z, nll, y_logits = self.forward(x, None)
            losses = compute_loss(nll, reduction="none")

        return {"val_loss": losses["total_loss"]}

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        images = self.sample()
        print("returning images")
        return {
            'val_loss': val_loss,
            "log": {"images": [wandb.Image(images, caption="samples")],
                    "val_loss":val_loss},
        }
