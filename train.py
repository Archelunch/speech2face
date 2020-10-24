import argparse
import hydra
import os
import json
import shutil
import random
from itertools import islice
import yaml

import torch
import torch.nn.functional as F

# from torch.optim.swa_utils import AveragedModel, SWALR
import torch.optim as optim
import torch.utils.data as data
import torch.autograd.profiler as profiler

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from datasets import get_CIFAR10, get_SVHN, get_CELEBA, postprocess
from model import Glow
from mintnet_model import Net

from torchvision.utils import make_grid

from lightning_module import GlowLighting
from mintnet_lightning import MintLighting


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


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
    saved_checkpoint = cfg.saved_checkpoint
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
    warmup = cfg.warmup
    precision = cfg.precision
    num_gpu = cfg.num_gpu
    accumulate_grad_batches = cfg.accumulate_grad_batches
    db = cfg.db
    num_nodes = cfg.num_nodes
    #mintnet_config = cfg.mintnet_config

    os.environ['WANDB_API_KEY'] = cfg.wandb_key
    os.environ['WANDB_MODE'] = "dryrun"

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

    def init_act():
        print("Started init")
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
        print("Started loop")
        with torch.no_grad():
            for batch, target in islice(train_loader, None, n_init_batches):
                init_batches.append(batch)
                init_targets.append(target)
            print("Finished loop")
            init_batches = torch.cat(init_batches).cuda()
            assert init_batches.shape[0] == n_init_batches * batch_size

            if y_condition:
                init_targets = torch.cat(init_targets).cuda()
            else:
                init_targets = None

            temp = model.forward(init_batches, init_targets)
            print("Finished initialization")
        del init_batches
        del init_targets
        del train_loader
        del temp
        model.cpu()
        torch.cuda.empty_cache()

    init_act()

    glow_light = GlowLighting(
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
    )

    wandb_logger = WandbLogger(
        name="Back to GLOW experiment with CELEBA", project="glow-experiments"
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=output_dir,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=num_gpu,
        num_nodes=num_nodes,
        distributed_backend=db,
        gradient_clip_val=max_grad_norm,
        logger=wandb_logger,
        precision=precision,
        checkpoint_callback=checkpoint_callback,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=0.2,
        resume_from_checkpoint=saved_checkpoint,
        auto_select_gpus=True,
    )
    trainer.fit(glow_light)


if __name__ == "__main__":
    main()
