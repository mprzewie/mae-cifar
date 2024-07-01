import os
from typing import Tuple, Literal

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, STL10
from torchvision.transforms import ToTensor, Compose, Normalize, transforms

def get_datasets(args, *, stl_train_ctx: Literal["train", "test"] = None) -> Tuple[Dataset, Dataset, dict]:
    if args.ds == "cifar10":
        root = args.named_ds_root / "cifar10"
        train_dataset = torchvision.datasets.CIFAR10(root, train=True, download=True,
                                                     transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
        val_dataset = torchvision.datasets.CIFAR10(root, train=False, download=True,
                                                   transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))

    elif args.ds == "cifar100":
        root = args.named_ds_root / "cifar100"
        train_dataset = torchvision.datasets.CIFAR100(root, train=True, download=True,
                                                      transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
        val_dataset = torchvision.datasets.CIFAR100(root, train=False, download=True,
                                                    transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))

    elif args.ds == "stl10":
        root = args.named_ds_root / "stl10"

        mean = torch.tensor([0.43, 0.42, 0.39])
        std = torch.tensor([0.27, 0.26, 0.27])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.resolution, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std, )
        ])
        transform_val = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std, )
        ])
        if stl_train_ctx == "train":
            train_split = "train+unlabeled"
        elif stl_train_ctx == "test":
            train_split = "train"
        else:
            assert False, f"{stl_train_ctx=}"

        train_dataset = STL10(root, split=train_split, transform=transform_train, download=True)
        val_dataset = STL10(root, split='test', transform=transform_val, download=True)

    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.resolution, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([
            transforms.Resize((args.resolution * (16/14)), interpolation=3),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataset = ImageFolder(os.path.join(args.ds, 'train'), transform=transform_train)
        val_dataset = ImageFolder(os.path.join(args.ds, 'val'), transform=transform_val)


    return train_dataset, val_dataset #, imsize_kwargs


def parse_ds_args(args):
    if args.resolution is not None and "cifar" in args.ds:
        assert False, "Not allowed to change resolution with cifar"

    if args.resolution is None:
        if args.ds == "stl10":
            args.resolution = 96
        elif "cifar" not in args.ds:
            args.resolution = 224
        else:
            args.resolution = 32

    assert args.resolution % 16 == 0
    args.patch_size = args.resolution // 16

