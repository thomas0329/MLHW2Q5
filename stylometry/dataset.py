# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .sgf_go import load_player_games_tokens
from .boards import load_player_games_planes


class GE2EDataset(Dataset):
    """Lightweight index over per-player files for GE2E sampling."""

    def __init__(self, root_dir: Path, seq_len: int = 64, max_games_per_file: int = 256):
        self.root_dir = Path(root_dir)
        self.seq_len = seq_len
        self.max_games_per_file = max_games_per_file
        self.files: List[Path] = sorted(self.root_dir.glob("*.sgf"), key=lambda p: p.name)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray, np.ndarray]:
        p = self.files[idx]
        player_id = p.stem
        toks, msk = load_player_games_tokens(p, seq_len=self.seq_len, max_games=self.max_games_per_file)
        return player_id, toks, msk


def sample_ge2e_batch(
        dataset: GE2EDataset,
        players_per_batch: int,
        games_per_player: int,
        rng: np.random.RandomState,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Sample a GE2E batch: N players × M games, each seq_len tokens.
    Returns:
    - tokens: [N*M, L] int64
    - masks:  [N*M, L] bool
    - player_indices: list length N indicating which dataset indices were used
    """
    N = players_per_batch
    M = games_per_player
    L = dataset.seq_len

    player_indices = rng.choice(len(dataset), size=N, replace=False).tolist()
    all_tokens: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []

    for pi in player_indices:
        _, toks, msk = dataset[pi]
        G = toks.shape[0]
        if G == 0:
            # fallback: single PASS row
            sel_tok = np.full((M, L), fill_value=361, dtype=np.int64)
            sel_msk = np.zeros((M, L), dtype=np.bool_)
        else:
            if G >= M:
                idxs = rng.choice(G, size=M, replace=False)
                sel_tok = toks[idxs]
                sel_msk = msk[idxs]
            else:
                # sample with replacement to reach M
                idxs = rng.choice(G, size=M, replace=True)
                sel_tok = toks[idxs]
                sel_msk = msk[idxs]
        all_tokens.append(sel_tok)
        all_masks.append(sel_msk)

    tokens_np = np.concatenate(all_tokens, axis=0)
    masks_np = np.concatenate(all_masks, axis=0)
    return (
        torch.from_numpy(tokens_np.astype(np.int64)),
        torch.from_numpy(masks_np.astype(np.bool_)),
        player_indices,
    )


class GE2EBoardDataset(Dataset):
    """Dataset indexing per-player SGF files to return board planes sequences."""

    def __init__(self, root_dir: Path, seq_len: int = 64, max_games_per_file: int = 128):
        self.root_dir = Path(root_dir)
        self.seq_len = seq_len
        self.max_games_per_file = max_games_per_file
        self.files: List[Path] = sorted(self.root_dir.glob("*.sgf"), key=lambda p: p.name)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray, np.ndarray]:
        p = self.files[idx]
        player_id = p.stem
        planes, mask = load_player_games_planes(p, seq_len=self.seq_len, max_games=self.max_games_per_file)
        return player_id, planes, mask


def sample_ge2e_board_batch(
    dataset: GE2EBoardDataset,
    players_per_batch: int,
    games_per_player: int,
    rng: np.random.RandomState,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Sample a GE2E batch returning boards: [N*M, T, C, H, W] and mask [N*M, T]."""
    N = players_per_batch
    M = games_per_player
    T = dataset.seq_len
    player_indices = rng.choice(len(dataset), size=N, replace=False).tolist()
    all_boards: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    for pi in player_indices:
        _, planes, mask = dataset[pi]
        G = planes.shape[0]
        if G == 0:
            boards_sel = np.zeros((M, T, 4, 19, 19), dtype=np.float32)
            mask_sel = np.zeros((M, T), dtype=np.bool_)
        else:
            if G >= M:
                idxs = rng.choice(G, size=M, replace=False)
                boards_sel = planes[idxs]
                mask_sel = mask[idxs]
            else:
                idxs = rng.choice(G, size=M, replace=True)
                boards_sel = planes[idxs]
                mask_sel = mask[idxs]
        all_boards.append(boards_sel)
        all_masks.append(mask_sel)

    boards_np = np.concatenate(all_boards, axis=0)
    masks_np = np.concatenate(all_masks, axis=0)
    return (
        torch.from_numpy(boards_np.astype(np.float32)),
        torch.from_numpy(masks_np.astype(np.bool_)),
        player_indices,
    )
