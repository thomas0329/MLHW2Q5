# -*- coding: utf-8 -*-
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Reuse SGF parsing patterns similar to sgf_go
SGF_MOVE_RE = re.compile(r";([BW])\[([^\]]*)\]")
SGF_PLAYER_RE = re.compile(r"\bP([BW])\[([^\]]+)\]")
SGF_GAME_SPLIT_RE = re.compile(r"\((?:[^()]*|\([^()]*\))*\)")

ALPHA = [c for c in "abcdefghijklmnopqrstuvwxyz" if c != "i"]
LETTER_TO_IDX = {c: i for i, c in enumerate(ALPHA)}
BOARD_SIZE = 19


def _sgf_coord_to_xy(coord: str) -> Optional[Tuple[int, int]]:
	if coord == "":
		return None
	if len(coord) != 2:
		return None
	cx, cy = coord[0], coord[1]
	if cx not in LETTER_TO_IDX or cy not in LETTER_TO_IDX:
		return None
	x = LETTER_TO_IDX[cx]
	y = LETTER_TO_IDX[cy]
	return x, y


def _extract_games(sgf_text: str) -> List[str]:
	return SGF_GAME_SPLIT_RE.findall(sgf_text)


def _detect_player_color(game_text: str, player_id: str) -> Optional[str]:
	pbs = SGF_PLAYER_RE.findall(game_text)
	for c, name in pbs:
		if name.strip() == player_id.strip():
			return c
	return None


def _first_non_empty_line(text: str) -> str:
	for line in text.splitlines():
		line = line.strip()
		if line:
			return line
	return ""


def _planes_from_board(black: np.ndarray, white: np.ndarray, last_xy: Optional[Tuple[int, int]], side_to_move: int) -> np.ndarray:
	# Channels: [0]=black stones, [1]=white stones, [2]=last-move marker, [3]=side-to-move full plane
	planes = np.zeros((4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
	planes[0] = black
	planes[1] = white
	if last_xy is not None:
		x, y = last_xy
		planes[2, y, x] = 1.0
	planes[3, :, :] = 1.0 if side_to_move > 0 else 0.0
	return planes


def _apply_move(black: np.ndarray, white: np.ndarray, color: str, xy: Optional[Tuple[int, int]]) -> None:
	# Very simple apply: place stone; skip captures/ko handling (data assumed legal)
	if xy is None:
		return
	x, y = xy
	if color == 'B':
		black[y, x] = 1.0
		white[y, x] = 0.0
	else:
		white[y, x] = 1.0
		black[y, x] = 0.0


def load_player_games_planes(
	file_path: Path,
	seq_len: int = 64,
	max_games: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Return (planes, mask) for target player's moves per game.
	planes: [G, T, C, 19, 19], mask: [G, T]
	"""
	text = file_path.read_text(encoding="utf-8", errors="ignore")
	player_id = _first_non_empty_line(text)
	games = _extract_games(text)

	planes_list: List[np.ndarray] = []
	mask_list: List[np.ndarray] = []

	for g in games:
		color = _detect_player_color(g, player_id)
		if color is None:
			continue
		moves = SGF_MOVE_RE.findall(g)
		# set up empty board
		black = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
		white = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
		last_xy: Optional[Tuple[int, int]] = None
		side_to_move = 1 if (len(moves) > 0 and moves[0][0] == 'B') else 0

		frames: List[np.ndarray] = []
		for mv_color, coord in moves:
			# if it's the target player's turn, record pre-move planes
			if mv_color == color:
				frames.append(_planes_from_board(black, white, last_xy, 1 if mv_color == 'B' else 0))
				xy = _sgf_coord_to_xy(coord)
				_apply_move(black, white, mv_color, xy)
				last_xy = xy
				side_to_move = 0 if mv_color == 'B' else 1
			else:
				xy = _sgf_coord_to_xy(coord)
				_apply_move(black, white, mv_color, xy)
				last_xy = xy
				side_to_move = 0 if mv_color == 'B' else 1

		if not frames:
			continue
		# pad/truncate to seq_len
		T = seq_len
		arr = np.zeros((T, 4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
		mask = np.zeros((T,), dtype=np.bool_)
		L = min(len(frames), T)
		arr[:L] = np.stack(frames[:L], axis=0)
		mask[:L] = True
		planes_list.append(arr)
		mask_list.append(mask)
		if max_games is not None and len(planes_list) >= max_games:
			break

	if not planes_list:
		return (
			np.zeros((0, seq_len, 4, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
			np.zeros((0, seq_len), dtype=np.bool_),
		)
	return np.stack(planes_list, axis=0), np.stack(mask_list, axis=0)
