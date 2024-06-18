import os
import argparse
import math
from pathlib import Path

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Normalize, transforms
from tqdm import tqdm

from model import *
from utils import setup_seed, maybe_setup_wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--logdir', type=Path)
    parser.add_argument("--linprobe", action="store_true")
    parser.add_argument("--arch", type=str, default="vit_tiny", choices=["vit_tiny", "vit_base"])
    parser.add_argument("--ds", default="cifar10", type=str)

    args = parser.parse_args()

    setup_seed(args.seed)

    maybe_setup_wandb(logdir=args.logdir, args=args, job_type="cls_cls_attention")


    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    if args.ds == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
        val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
        imsize_kwargs = dict(
            image_size=32,
            patch_size=2,
        )
    elif args.ds == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100('data', train=True, download=True,
                                                     transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
        val_dataset = torchvision.datasets.CIFAR100('data', train=False, download=True,
                                                   transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
        imsize_kwargs = dict(
            image_size=32,
            patch_size=2,
        )
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataset = ImageFolder(os.path.join(args.ds, 'train'), transform=transform_train)
        val_dataset = ImageFolder(os.path.join(args.ds, 'val'), transform=transform_val)
        imsize_kwargs = dict(
            image_size=224,
            patch_size=16,
        )
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vit_kwargs = VIT_KWARGS[args.arch]
    model = MAE_ViT(**vit_kwargs, **imsize_kwargs)
    ckpt = torch.load(args.logdir / f"{args.arch}-mae.pt", map_location='cpu')
    model.load_state_dict(ckpt["model"])
    writer = SummaryWriter(args.logdir)

    model = model.encoder

    for img, label in tqdm(iter(val_dataloader)):
        img = img.to(device)
        label = label.to(device)
        logits = model(img, return_attn_masks = True)

        assert False, logits.shape



