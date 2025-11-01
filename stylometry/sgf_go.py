# -*- coding: utf-8 -*-
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


SGF_MOVE_RE = re.compile(r";([BW])\[([^\]]*)\]")
SGF_PLAYER_RE = re.compile(r"\bP([BW])\[([^\]]+)\]")
SGF_GAME_SPLIT_RE = re.compile(r"\((?:[^()]*|\([^()]*\))*\)")

# SGF coordinates omit the letter 'i'. For 19x19 board.
ALPHA = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != "i"]
LETTER_TO_IDX = {c: i for i, c in enumerate(ALPHA)}
BOARD_SIZE = 19
NUM_POINTS = BOARD_SIZE * BOARD_SIZE
PASS_TOKEN = NUM_POINTS  # 361


def _sgf_coord_to_token(coord: str) -> int:
    """Convert SGF coord like 'pd' to flattened index, or PASS if empty.
    SGF pass is empty string. Coordinates are (col,row) with letters, without 'i'.
    """
    if coord == "":
        return PASS_TOKEN
    if len(coord) != 2:
        return PASS_TOKEN
    cx, cy = coord[0], coord[1]
    if cx not in LETTER_TO_IDX or cy not in LETTER_TO_IDX:
        return PASS_TOKEN
    x = LETTER_TO_IDX[cx]
    y = LETTER_TO_IDX[cy]
    # Flatten row-major (y, x)
    return int(y * BOARD_SIZE + x)


def _extract_games(sgf_text: str) -> List[str]:
    """Split concatenated SGF content into per-game strings."""
    return SGF_GAME_SPLIT_RE.findall(sgf_text)


def _detect_player_color(game_text: str, player_id: str) -> Optional[str]:
    """Return 'B' or 'W' if player_id matches PB/PW tags; otherwise None."""
    pbs = SGF_PLAYER_RE.findall(game_text)
    # pbs is list of tuples [("B","name"), ("W","name")]
    color = None
    for c, name in pbs:
        if name.strip() == player_id.strip():
            color = c
            break
    return color


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def load_player_games_tokens(
        file_path: Path,
        seq_len: int = 64,
        max_games: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a player .sgf file and return (tokens, mask).

    - tokens: shape [G, seq_len] integers in [0..361]
    - mask: shape [G, seq_len] booleans, True for valid timesteps
    We extract only moves made by the player (B or W depending on PB/PW).
    Games where the player name cannot be matched are skipped.
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    player_id = _first_non_empty_line(text)
    games = _extract_games(text)

    tokens_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []

    for g in games:
        color = _detect_player_color(g, player_id)
        if color is None:
            continue
        moves = SGF_MOVE_RE.findall(g)
        # Collect only moves of this player's color, in order
        seq: List[int] = []
        for mv_color, coord in moves:
            if mv_color == color:
                seq.append(_sgf_coord_to_token(coord))
        if len(seq) == 0:
            continue
        # pad / truncate
        arr = np.full((seq_len,), PASS_TOKEN, dtype=np.int64)
        mask = np.zeros((seq_len,), dtype=np.bool_)
        L = min(len(seq), seq_len)
        arr[:L] = np.asarray(seq[:L], dtype=np.int64)
        mask[:L] = True
        tokens_list.append(arr)
        mask_list.append(mask)
        if max_games is not None and len(tokens_list) >= max_games:
            break

    if not tokens_list:
        return (
            np.full((0, seq_len), PASS_TOKEN, dtype=np.int64),
            np.zeros((0, seq_len), dtype=np.bool_),
        )

    return np.stack(tokens_list, axis=0), np.stack(mask_list, axis=0)


def load_all_game_tokens_from_dir(
        dir_path: Path,
        seq_len: int = 64,
        max_games_per_file: Optional[int] = None,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Iterate .sgf files in a directory and load token sequences.
    Returns list of tuples: (player_file_stem, tokens[G,L], mask[G,L]).
    """
    out: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for p in sorted(dir_path.glob("*.sgf"), key=lambda x: x.name):
        toks, msk = load_player_games_tokens(p, seq_len=seq_len, max_games=max_games_per_file)
        out.append((p.stem, toks, msk))
    return out
