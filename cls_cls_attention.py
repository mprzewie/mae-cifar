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

from datasets import get_datasets, parse_ds_args
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
    parser.add_argument("--named_ds_root", default=Path("/shared/sets/datasets/vision/"), type=Path)
    parser.add_argument("--resolution", "--res", default=None, type=int)

    args = parser.parse_args()
    parse_ds_args(args)

    setup_seed(args.seed)

    maybe_setup_wandb(logdir=args.logdir, args=args, job_type="cls_cls_attention")


    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset, val_dataset = get_datasets(args, stl_train_ctx="test")

    # train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vit_kwargs = VIT_KWARGS[args.arch]
    model = MAE_ViT(**vit_kwargs, image_size=args.resolution, patch_size=args.patch_size)
    try:
        ckpt = torch.load(args.logdir / f"{args.arch}-mae.pt", map_location='cpu')
    except:
        ckpt = torch.load(args.logdir / f"vit-t-mae.pt", map_location='cpu')

    model.load_state_dict(ckpt["model"], strict=False)
    writer = SummaryWriter(args.logdir)

    model = model.encoder.to(device)

    cc_attns = []
    with torch.no_grad():
        for img, _ in tqdm(iter(val_dataloader)):
            img = img.to(device)
            _, _, attn = model(img, return_attn_masks = True, mask_ratio=0)

            cc_attn = attn[:, :, :, 0, 0].cpu().numpy()
            # B, L, H
            cc_attns.append(cc_attn)

    cc_attns = np.concatenate(cc_attns, axis=0)

    cc_attns = cc_attns.mean(axis=(0,2))

    for i, a in enumerate(cc_attns):
        writer.add_scalar("eval_attention/cls_cls", a, global_step=i)

    print(cc_attns)

