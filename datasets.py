from pathlib import Path

import torch
import torch.nn.functional as F

from torchvision import transforms, datasets

n_bits = 5


def preprocess(x):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

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


def get_CIFAR10(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])

    train_transform = transforms.Compose(transformations)

    def one_hot_encode(target): return F.one_hot(
        torch.tensor(target), num_classes)

    path = Path(dataroot) / "data" / "CIFAR10"
    train_dataset = datasets.CIFAR10(
        path,
        train=True,
        transform=train_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.CIFAR10(
        path,
        train=False,
        transform=test_transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset


def get_SVHN(augment, dataroot, download):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    transform = transforms.Compose(transformations)

    def one_hot_encode(target): return F.one_hot(
        torch.tensor(target), num_classes)

    path = Path(dataroot) / "data" / "SVHN"
    train_dataset = datasets.SVHN(
        path,
        split="train",
        transform=transform,
        target_transform=one_hot_encode,
        download=download,
    )

    test_dataset = datasets.SVHN(
        path,
        split="test",
        transform=transform,
        target_transform=one_hot_encode,
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset


def get_CELEBA(augment, dataroot, download):
    file_list = [
        # File ID                         MD5 Hash                            Filename
        ("1Lkz2FpYaopWqDt-v6M8H8DIB1I1AGCzX",
         "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        ("1a6ce6Z1uSh48ACJhh17J1YEyGTDT9nEV",
         "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        ("1If6Tlr0TkIlb-fsRVEr5HPZyMb_aTzYS",
         "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        ("1uj3mrD6FAEuhbmguGApfefqrQFztc0S4",
         "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        ("13XvEQSgQNhJJOCWA49ybhmhzEdqbbCfT", "cc24ecafdb5b50baae59b03474781f8c",
         "list_landmarks_align_celeba.txt"),
        ("1h9xZ5NzHP2mNRPNXcXi2WK11YVKV5GNN",
         "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]
    datasets.CelebA.file_list = file_list
    image_shape = (128, 128, 3)
    num_classes = 40
    test_transform = transforms.Compose(
        [transforms.Resize((128, 128)), transforms.ToTensor(), preprocess])

    if augment:
        transformations = [
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transformations = []
    transformations.extend(
        [transforms.Resize((128, 128)), transforms.ToTensor(), preprocess])

    train_transform = transforms.Compose(transformations)
    path = Path(dataroot) / "data" / "CELEBA"
    train_dataset = datasets.CelebA(
        path,
        split="train",
        transform=train_transform,
        target_type="attr",
        download=download,
    )

    test_dataset = datasets.CelebA(
        path,
        split="test",
        transform=test_transform,
        target_type="attr",
        download=download,
    )

    return image_shape, num_classes, train_dataset, test_dataset
