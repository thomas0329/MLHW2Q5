# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import time
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from stylometry.dataset import GE2EDataset, GE2EBoardDataset, sample_ge2e_batch, sample_ge2e_board_batch
from stylometry.model import TokenTransformerEncoder, BoardStylometryModel
from stylometry.ge2e_loss import GE2ELoss
from stylometry.sgf_go import load_player_games_tokens
from stylometry.boards import load_player_games_planes


def _norm_id(stem: str) -> str:
    """Normalize file stem to numeric id string."""
    m = re.search(r"\d+", stem)
    return str(int(m.group())) if m else stem


def embed_games_token(model: TokenTransformerEncoder, toks: np.ndarray, msk: np.ndarray, device: torch.device) -> np.ndarray:
    """Embed games and return mean embedding. toks: [G, L], msk: [G, L]."""
    if toks.shape[0] == 0:
        return np.zeros((model.proj[-1].out_features,), dtype=np.float32)
    with torch.no_grad():
        t = torch.from_numpy(toks).to(device)
        m = torch.from_numpy(msk).to(device)
        emb = model(t, m)  # [G, D]
        return emb.mean(dim=0).cpu().numpy().astype(np.float32)


def embed_games_board(model: BoardStylometryModel, planes: np.ndarray, mask: np.ndarray, device: torch.device) -> np.ndarray:
    """Embed games and return mean embedding. planes: [G, T, C, H, W], mask: [G, T]."""
    if planes.shape[0] == 0:
        return np.zeros((model.game_enc.proj[-1].out_features,), dtype=np.float32)
    with torch.no_grad():
        t_planes = torch.from_numpy(planes).to(device)
        t_mask = torch.from_numpy(mask).to(device)
        vecs = model(t_planes, t_mask)  # [G, D]
        return vecs.mean(dim=0).cpu().numpy().astype(np.float32)


def evaluate_train_accuracy(
    model: nn.Module,
    train_dir: Path,
    device: torch.device,
    seq_len: int,
    arch: str,
    max_players: Optional[int] = None,
) -> float:
    """Evaluate accuracy on training data using leave-one-game-out validation."""
    model.eval()
    files = sorted(train_dir.glob("*.sgf"), key=lambda p: p.name)
    if max_players is not None:
        files = files[:max_players]

    # Precompute base prototypes for all players (for other players, use all games)
    print('  Building player prototypes...', end=' ', flush=True)
    base_prototypes: Dict[str, np.ndarray] = {}
    for file_path in files:
        player_id = file_path.stem
        if arch == 'board':
            planes, mask = load_player_games_planes(file_path, seq_len=seq_len)
            if planes.shape[0] == 0:
                continue
            emb = embed_games_board(model, planes, mask, device)
        else:
            toks, msk = load_player_games_tokens(file_path, seq_len=seq_len)
            if toks.shape[0] == 0:
                continue
            emb = embed_games_token(model, toks, msk, device)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        base_prototypes[player_id] = emb
    print(f'done ({len(base_prototypes)} players)')

    if len(base_prototypes) < 2:
        model.train()
        return 0.0

    player_ids = sorted(base_prototypes.keys())

    correct = 0
    total = 0

    # Evaluate each player by holding out one game and comparing against all prototypes
    for file_path in tqdm(files, desc='  Evaluating queries', leave=False):
        player_id = file_path.stem
        if player_id not in base_prototypes:
            continue

        if arch == 'board':
            planes, mask = load_player_games_planes(file_path, seq_len=seq_len)
            G = planes.shape[0]
            if G < 2:  # Need at least 2 games to hold one out
                continue
            # Hold out last game for query, use rest for prototype
            query_planes = planes[-1:]  # [1, T, C, H, W]
            query_mask = mask[-1:]  # [1, T]
            prototype_planes = planes[:-1]  # [G-1, T, C, H, W]
            prototype_mask = mask[:-1]  # [G-1, T]

            query_emb = embed_games_board(model, query_planes, query_mask, device)
            query_player_proto = embed_games_board(model, prototype_planes, prototype_mask, device)
        else:
            toks, msk = load_player_games_tokens(file_path, seq_len=seq_len)
            G = toks.shape[0]
            if G < 2:  # Need at least 2 games to hold one out
                continue
            # Hold out last game for query, use rest for prototype
            query_toks = toks[-1:]  # [1, L]
            query_msk = msk[-1:]  # [1, L]
            prototype_toks = toks[:-1]  # [G-1, L]
            prototype_msk = msk[:-1]  # [G-1, L]

            query_emb = embed_games_token(model, query_toks, query_msk, device)
            query_player_proto = embed_games_token(model, prototype_toks, prototype_msk, device)

        # Normalize embeddings
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-12)
        query_player_proto = query_player_proto / (np.linalg.norm(query_player_proto) + 1e-12)

        # Build prototype matrix: use updated prototype for query player, base for others
        prototypes_list = []
        for pid in player_ids:
            if pid == player_id:
                prototypes_list.append(query_player_proto)
            else:
                prototypes_list.append(base_prototypes[pid])
        proto_matrix = np.stack(prototypes_list, axis=0)  # [N, D]

        # Compute similarities to all prototypes
        sims = proto_matrix @ query_emb  # [N]
        best_idx = int(np.argmax(sims))
        best_id = player_ids[best_idx]

        # Check if correct player is predicted
        if best_id == player_id:
            correct += 1
        total += 1

    model.train()
    return correct / total if total > 0 else 0.0


def evaluate_test_accuracy(
    model: nn.Module,
    test_root: Path,
    device: torch.device,
    seq_len: int,
    arch: str,
) -> Optional[float]:
    """Evaluate accuracy on test data. Returns None if ground truth not available."""
    model.eval()

    # Find candidate and query directories
    cand_dir = None
    query_dir = None
    for name in ['cand_set', 'candidate', 'candidates']:
        candidate = test_root / name
        if candidate.exists() and candidate.is_dir():
            if len(list(candidate.glob("*.sgf"))) > 0:
                cand_dir = candidate
                break
    for name in ['query_set', 'query', 'queries']:
        candidate = test_root / name
        if candidate.exists() and candidate.is_dir():
            if len(list(candidate.glob("*.sgf"))) > 0:
                query_dir = candidate
                break

    if cand_dir is None or query_dir is None:
        return None

    # Build candidate prototypes
    cand_prototypes: Dict[str, np.ndarray] = {}
    cand_files = sorted(cand_dir.glob("*.sgf"), key=lambda x: x.name)
    for p in tqdm(cand_files, desc='Loading candidates', leave=False):
        cid = _norm_id(p.stem)
        if arch == 'board':
            planes, mask = load_player_games_planes(p, seq_len=seq_len)
            if planes.shape[0] == 0:
                continue
            emb = embed_games_board(model, planes, mask, device)
        else:
            toks, msk = load_player_games_tokens(p, seq_len=seq_len)
            if toks.shape[0] == 0:
                continue
            emb = embed_games_token(model, toks, msk, device)
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        cand_prototypes[cid] = emb

    if len(cand_prototypes) == 0:
        return None

    cand_ids = sorted(cand_prototypes.keys(), key=lambda s: int(s) if s.isdigit() else s)
    C = np.stack([cand_prototypes[c] for c in cand_ids], axis=0)  # [Nc, D]

    # Evaluate queries
    query_files = sorted(query_dir.glob("*.sgf"), key=lambda x: x.name)
    correct = 0
    total = 0

    for p in tqdm(query_files, desc='Evaluating test', leave=False):
        qid = _norm_id(p.stem)
        if arch == 'board':
            planes, mask = load_player_games_planes(p, seq_len=seq_len)
            if planes.shape[0] == 0:
                continue
            q = embed_games_board(model, planes, mask, device)
        else:
            toks, msk = load_player_games_tokens(p, seq_len=seq_len)
            if toks.shape[0] == 0:
                continue
            q = embed_games_token(model, toks, msk, device)
        q = q / (np.linalg.norm(q) + 1e-12)

        # Find best candidate
        sims = (C @ q)
        best_idx = int(np.argmax(sims))
        pred_cid = cand_ids[best_idx]

        # Check if ground truth is available (query ID matches a candidate ID)
        if qid in cand_prototypes:
            if pred_cid == qid:
                correct += 1
            total += 1

    model.train()
    return correct / total if total > 0 else None


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
    p.add_argument('--arch', type=str, default='token', choices=['token','board'])
    p.add_argument('--eval-interval', type=int, default=1000, help='Evaluate every N steps')
    p.add_argument('--eval-train-max-players', type=int, default=100, help='Max players to evaluate on train set')
    p.add_argument('--save-interval', type=int, default=5000, help='Save checkpoint every N steps (0 to disable)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.RandomState(args.seed)

    train_dir = Path(args.data_root) / 'train_set'
    if args.arch == 'board':
        ds = GE2EBoardDataset(train_dir, seq_len=args.seq_len)
        model = BoardStylometryModel()
    else:
        ds = GE2EDataset(train_dir, seq_len=args.seq_len)
        model = TokenTransformerEncoder()
    model.to(device)
    criterion = GE2ELoss().to(device)
    opt = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=args.lr)
    scaler = GradScaler(enabled=bool(args.amp))
    writer = SummaryWriter(log_dir=args.logdir)

    # Create unique checkpoint directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(args.save)
    checkpoint_dir = save_path.parent / f"{save_path.stem}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = checkpoint_dir / save_path.name
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    model.train()
    start = time.time()
    test_root = Path(args.data_root) / 'test_set'
    pbar = tqdm(range(1, args.steps + 1), desc='Training')
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        N = args.players_per_batch
        M = args.games_per_player
        if args.arch == 'board':
            boards, msk, player_idx = sample_ge2e_board_batch(ds, N, M, rng)
            boards = boards.to(device)
            msk = msk.to(device)
            with autocast(enabled=bool(args.amp)):
                emb = model(boards, msk)  # [B, D]
                loss = criterion(emb, N, M)
        else:
            tok, msk, player_idx = sample_ge2e_batch(ds, N, M, rng)
            tok = tok.to(device)
            msk = msk.to(device)
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

        # Periodic evaluation
        if step % args.eval_interval == 0 or step == args.steps:
            print(f'\n[Step {step}] Evaluating...')
            # Evaluate on training data
            train_acc = evaluate_train_accuracy(
                model, train_dir, device, args.seq_len, args.arch,
                max_players=args.eval_train_max_players
            )
            writer.add_scalar('eval/train_acc', train_acc, step)
            print(f'  Train accuracy: {train_acc:.4f}')

            # Evaluate on test data (if available)
            test_acc = evaluate_test_accuracy(
                model, test_root, device, args.seq_len, args.arch
            )
            if test_acc is not None:
                writer.add_scalar('eval/test_acc', test_acc, step)
                print(f'  Test accuracy: {test_acc:.4f}')
            else:
                print(f'  Test accuracy: N/A (ground truth not available)')
            print()
            model.train()  # Switch back to training mode

        # Periodic checkpoint saving
        if args.save_interval > 0 and (step % args.save_interval == 0):
            checkpoint_path = checkpoint_dir / f"checkpoint_step{step}{save_path.suffix}"
            torch.save({'model': model.state_dict(), 'crit': criterion.state_dict(), 'step': step}, checkpoint_path)
            print(f"Saved intermediate checkpoint to {checkpoint_path}")

    # save final checkpoint
    torch.save({'model': model.state_dict(), 'crit': criterion.state_dict(), 'step': args.steps}, final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")
    writer.flush()
    writer.close()


if __name__ == '__main__':
    main()
