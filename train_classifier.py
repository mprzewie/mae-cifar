import os
import argparse
import math
from pathlib import Path

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder, STL10
from torchvision.transforms import ToTensor, Compose, Normalize, transforms
from tqdm import tqdm

from datasets import get_datasets
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
    parser.add_argument("--num_last_blocks", "-nlb", type=int, default=1)


    args = parser.parse_args()

    setup_seed(args.seed)

    maybe_setup_wandb(logdir=args.logdir, args=args, job_type=("linprobe" if args.linprobe else "finetune"))


    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset, val_dataset, imsize_kwargs = get_datasets(args, stl_train_ctx="test")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vit_kwargs = VIT_KWARGS[args.arch]
    model = MAE_ViT(**vit_kwargs, **imsize_kwargs)

    ckpt = torch.load(args.logdir / f"{args.arch}-mae.pt", map_location='cpu')
    model.load_state_dict(ckpt["model"])
    writer = SummaryWriter(args.logdir)
    model = ViT_Classifier(
        model.encoder, num_classes=(10 if args.ds=="cifar10" else 100 if args.ds=="cifar100" else 1000),
        linprobe=args.linprobe,
        num_last_blocks=args.num_last_blocks
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    if args.linprobe:
        optim = torch.optim.AdamW(model.head.parameters(),lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    best_val_acc = 0
    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(val_dataloader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'best model with acc {best_val_acc} at {e} epoch!')
            # torch.save(model, args.output_model_path)

        writer.add_scalar('eval_loss/train', avg_train_loss, global_step=e)
        writer.add_scalar('eval_loss/val', avg_val_loss, global_step=e)
        writer.add_scalar('eval_accuracy/train', avg_train_acc, global_step=e)
        writer.add_scalar('eval_accuracy/val', avg_val_acc, global_step=e)
