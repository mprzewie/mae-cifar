import os
import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *
from utils import setup_seed, maybe_setup_wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument("--logdir", type=Path)
    parser.add_argument("--umae_lambda", type=float, default=0)
    parser.add_argument("--latent_lambda", type=float, default=0)
    parser.add_argument("--latent_loss_detach_targets", "-lldt", type=bool, action="store_true")

    args = parser.parse_args()

    setup_seed(args.seed)

    args.logdir.mkdir(parents=True, exist_ok=True)

    maybe_setup_wandb(logdir=args.logdir, args=args, job_type='pretrain')

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    writer = SummaryWriter(args.logdir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        metrics = defaultdict(list)

        for img, label in tqdm(iter(dataloader), desc=f"Pretrain: {e}"):
            step_count += 1
            img = img.to(device)
            predicted_img, mask, features, l_decoder_features = model(img)

            cls_features = features[0]

            # umae
            norm_features =  torch.nn.functional.normalize(cls_features)

            sim = norm_features @ norm_features.T
            loss_umae = sim.pow(2).mean()
            ####
            # latent decoder
            target_features = features[1:]
            if args.latent_loss_detach_targets:
                target_features = target_features.detach()
            loss_latent_decoder = ((features[1:] - l_decoder_features) ** 2).mean()
            ####

            loss_mae = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio

            loss = loss_mae + (args.umae_lambda * loss_umae) + (args.latent_lambda * loss_latent_decoder)

            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()

            metrics["loss_total"].append(loss.item())
            metrics["loss_mae"].append(loss_mae.item())
            metrics["loss_umae"].append(loss_umae.item())
            metrics["loss_latent"].append(loss_latent_decoder.item())

        for k, v in metrics.items():
            writer.add_scalar(f"train/{k}", np.mean(v), global_step=e)
        writer.add_scalar("train/epoch", e, global_step=e)
                
        writer.add_scalar("train/lr", lr_scheduler.get_lr()[-1], global_step=e)
        # writer.add_scalar("train/loss_total", loss.item(), global_step=e)
        lr_scheduler.step()
        # avg_loss = sum(losses) / len(losses)
        # writer.add_scalar('train_loss_mae', avg_loss, global_step=e)
        if e % 10 == 0:
            print(e, {k: np.mean(v) for k,v in metrics.items()})
        # print(f'In epoch {e}, average traning loss is {avg_loss}.')

        # ''' visualize the first 16 predicted images on val dataset'''
        # model.eval()
        #
        # with torch.no_grad():
        #     val_img = torch.stack([val_dataset[i][0] for i in range(16)])
        #     val_img = val_img.to(device)
        #     predicted_val_img, mask = model(val_img)
        #     predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
        #     img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
        #     img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
        #     # writer.add_image('mae_image', (img + 1) / 2, global_step=e)
        
        ''' save model '''
        ckpt = {
            "model": model.state_dict(),
            "epoch": e,
            "metrics": {k: np.mean(v) for k,v in metrics.items()}

        }
        torch.save(ckpt, args.logdir / "vit-t-mae.pt")