# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from stylometry.dataset import GE2EDataset, sample_ge2e_batch
from stylometry.model import TokenTransformerEncoder
from stylometry.ge2e_loss import GE2ELoss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train GE2E token-based Go stylometry model')
    p.add_argument('--data-root', type=str, required=True)
    p.add_argument('--save', type=str, required=True)
    p.add_argument('--seq-len', type=int, default=64)
    p.add_argument('--players-per-batch', type=int, default=32)
    p.add_argument('--games-per-player', type=int, default=6)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--amp', type=int, default=1)
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--logdir', type=str, default='/workspace/runs/ge2e')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.RandomState(args.seed)

    train_dir = Path(args.data_root) / 'train_set'
    ds = GE2EDataset(train_dir, seq_len=args.seq_len)

    model = TokenTransformerEncoder()
    model.to(device)
    criterion = GE2ELoss().to(device)
    opt = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=args.lr)
    scaler = GradScaler(enabled=bool(args.amp))
    writer = SummaryWriter(log_dir=args.logdir)

    model.train()
    start = time.time()
    pbar = tqdm(range(1, args.steps + 1), desc='Training')
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        tok, msk, player_idx = sample_ge2e_batch(
            ds, args.players_per_batch, args.games_per_player, rng
        )
        tok = tok.to(device)
        msk = msk.to(device)
        N = args.players_per_batch
        M = args.games_per_player
        with autocast(enabled=bool(args.amp)):
            emb = model(tok, msk)  # [N*M, D]
            loss = criterion(emb, N, M)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        # Logging
        writer.add_scalar('train/loss', float(loss.item()), step)
        writer.add_scalar('train/grad_norm', float(grad_norm), step)
        writer.add_scalar('train/w', float(criterion.w.detach().item()), step)
        writer.add_scalar('train/b', float(criterion.b.detach().item()), step)
        if step % 100 == 0:
            elapsed = time.time() - start
            steps_per_sec = step / max(elapsed, 1e-6)
            eta_sec = (args.steps - step) / max(steps_per_sec, 1e-6)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'steps/s': f'{steps_per_sec:.1f}',
                'eta': f'{eta_sec / 60:.1f}m'
            })

    # save checkpoint
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'crit': criterion.state_dict()}, save_path)
    print(f"Saved checkpoint to {save_path}")
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
