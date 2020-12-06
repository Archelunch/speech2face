import os
import random
import torch
from omegaconf import OmegaConf
from itertools import islice

# from torch.optim.swa_utils import AveragedModel, SWALR
import torch.utils.data as data


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from decoder_lighhtning import DecoderLighting
from datasets import get_dataset
from res_to_glow import Decoder as res_to_glow
from utils import get_frozen_glow


def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))


def main():
    cfg = OmegaConf.load("config.yaml")
    os.environ["WANDB_API_KEY"] = cfg.wandb_key
    os.environ["WANDB_MODE"] = "dryrun"

    check_manual_seed(cfg.env.seed)

    train_dataset, test_dataset = get_dataset(**cfg.dataset)
    frozen_glow = get_frozen_glow(cfg.glow)

    decoder_light = DecoderLighting(
        frozen_glow,
        res_to_glow,
        cfg.trani_params.lr,
        train_dataset,
        test_dataset,
        cfg.train_params.batch_size,
        cfg.train_params.eval_batch_size,
        cfg.train_params.n_workers,
    )

    wandb_logger = WandbLogger(name="RES TO GLOW", project="glow-experiments")

    checkpoint_callback = ModelCheckpoint(
        filepath=cfg.env.output_dir,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=cfg.train_params.epochs,
        gpus=cfg.env.num_gpu,
        num_nodes=cfg.env.num_nodes,
        distributed_backend=cfg.env.db,
        gradient_clip_val=cfg.train_params.max_grad_norm,
        logger=wandb_logger,
        precision=cfg.train_params.precision,
        checkpoint_callback=checkpoint_callback,
        accumulate_grad_batches=cfg.train_params.accumulate_grad_batches,
        val_check_interval=0.2,
        resume_from_checkpoint=cfg.env.saved_checkpoint,
        auto_select_gpus=True,
    )
    trainer.fit(decoder_light)


if __name__ == "__main__":
    main()
