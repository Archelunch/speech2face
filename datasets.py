import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from pathlib import Path

n_bits = 5


def preprocess(x):
    x = x * 255  # undo ToTensor scaling to [0,1]
    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5
    return x


def postprocess(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2 ** n_bits
    return torch.clamp(x, 0, 255).byte()


to_tensor = transforms.ToTensor()
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def process_one_image(img_path, size):
    img = Image.open(img_path)
    img = img.resize((size, size))
    img = to_tensor(img)
    # img = normalize(img)
    return preprocess(img)


class ImagesEmbeddingsDataset(Dataset):
    """Input data: list of dicts, path, label
    Returns: tensor, label"""

    def __init__(self, data_path, size):
        with open(os.path.abspath(data_path), "r") as f:
            files = f.read()

        data = [f.split(" ") for f in files.split("\n")]
        self.data = data
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, emb_path = self.data[idx]
        img = process_one_image(img_path, self.size)
        emb = torch.tensor(np.load(emb_path))  # .unsqueeze(0)
        # Dummy to tensor for code compatabilty
        return emb, preprocess(img)


def get_dataset(train_path, val_path, image_size=128):
    return ImagesEmbeddingsDataset(train_path, image_size), ImagesEmbeddingsDataset(
        val_path, image_size
    )
