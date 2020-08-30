import hydra
import os
import json
import shutil
import random
from itertools import islice

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
import torch.optim as optim
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from datasets import get_CIFAR10, get_SVHN, get_CELEBA
from model import Glow


def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))


def check_dataset(dataset, dataroot, augment, download):
    if dataset == "cifar10":
        cifar10 = get_CIFAR10(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = cifar10
    elif dataset == "svhn":
        svhn = get_SVHN(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = svhn
    elif dataset == "celeba":
        celeba = get_CELEBA(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = celeba
    return input_size, num_classes, train_dataset, test_dataset


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
    def __init__(self, model, opt_type, lr, train_dataset, test_dataset, batch_size, eval_batch_size, n_workers, use_swa, swa_lr, y_condition, y_weight, warmup, n_init_batches, multi_class=False):
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
        return self.model.forward(x=x, y_onehot=y_onehot, z=z, temperature=temperature, reverse=reverse)

    def configure_optimizers(self):
        """TODO SWA"""
        if self.opt_type == "AdamW":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(epoch): return min(1.0, (epoch + 1) / self.warmup)  # noqa

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)

        if self.use_swa:
            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(optimizer, swa_lr=self.swa_lr)

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
            drop_last=False,
        )
        return test_loader

    def training_step(self, batch, batch_nb):
        x, y = batch
        if self.y_condition:
            z, nll, y_logits = self.forward(x, y)
            losses = compute_loss_y(
                nll, y_logits, self.y_weight, y, self.multi_class)
        else:
            z, nll, y_logits = self.forward(x, None)
            losses = compute_loss(nll)

        return {
            'loss': losses["total_loss"],
            'log': {
                'train_loss': losses["total_loss"]
            }
        }

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
        return {
            'val_loss': losses["total_loss"],
        }


@hydra.main(config_path="config.yaml")
def main(cfg):

    dataset = cfg.dataset
    dataroot = cfg.dataroot
    download = cfg.download
    augment = cfg.augment
    batch_size = cfg.batch_size
    eval_batch_size = cfg.eval_batch_size
    epochs = cfg.epochs
    saved_model = cfg.saved_model
    seed = cfg.seed
    hidden_channels = cfg.hidden_channels
    K = cfg.K
    L = cfg.L
    actnorm_scale = cfg.actnorm_scale
    flow_permutation = cfg.flow_permutation
    flow_coupling = cfg.flow_coupling
    LU_decomposed = cfg.LU_decomposed
    learn_top = cfg.learn_top
    y_condition = cfg.y_condition
    y_weight = cfg.y_weight
    max_grad_clip = cfg.max_grad_clip
    max_grad_norm = cfg.max_grad_norm
    opt_type = cfg.opt_type
    lr = cfg.lr
    use_swa = cfg.use_swa
    swa_start = cfg.swa_start
    swa_lr = cfg.swa_lr
    n_workers = cfg.n_workers
    cuda = cfg.cuda
    n_init_batches = cfg.n_init_batches
    output_dir = cfg.output_dir
    saved_optimizer = cfg.saved_optimizer
    warmup = cfg.warmup
    precision = cfg.precision
    num_gpu = cfg.num_gpu
    accumulate_grad_batches = cfg.accumulate_grad_batches

    try:
        os.makedirs(cfg.output_dir)
    except FileExistsError:
        if cfg.fresh:
            shutil.rmtree(cfg.output_dir)
            os.makedirs(cfg.output_dir)
        if (not os.path.isdir(cfg.output_dir)) or (len(os.listdir(cfg.output_dir)) > 0):
            raise FileExistsError(
                "Please provide a path to a non-existing or empty directory. Alternatively, pass the --fresh flag."  # noqa
            )

    check_manual_seed(seed)

    ds = check_dataset(dataset, dataroot, augment, download)
    image_shape, num_classes, train_dataset, test_dataset = ds

    model = Glow(
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        num_classes,
        learn_top,
        y_condition,
    )
    # TODO move to method
    model.cuda()
    model.train()
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,
    )
    init_batches = []
    init_targets = []

    with torch.no_grad():
        for batch, target in islice(train_loader, None, n_init_batches):
            init_batches.append(batch)
            init_targets.append(target)

        init_batches = torch.cat(init_batches).cuda()

        print(init_batches.shape[0], n_init_batches * batch_size)
        assert init_batches.shape[0] == n_init_batches * batch_size

        if y_condition:
            init_targets = torch.cat(init_targets).cuda()
        else:
            init_targets = None

        model.forward(init_batches, init_targets)
        print("Finished initialization")
    del init_batches
    del init_targets
    del train_loader
    model.cpu()
    # END

    glow_light = GlowLighting(model, opt_type, lr, train_dataset, test_dataset,
                              batch_size, eval_batch_size, n_workers, use_swa, swa_lr, y_condition, y_weight, warmup, n_init_batches)

    wandb_logger = WandbLogger(
        name='Glow experiment with faces', project='glow-experiments')

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(output_dir, saved_model), verbose=True, monitor='val_loss', mode='min')
    trainer = pl.Trainer(max_epochs=epochs, gpus=num_gpu,
                         gradient_clip_val=max_grad_norm, logger=wandb_logger, precision=precision, checkpoint_callback=checkpoint_callback, accumulate_grad_batches=accumulate_grad_batches)
    trainer.fit(glow_light)


if __name__ == "__main__":
    main()
