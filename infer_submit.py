# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import torch
from stylometry.sgf_go import load_player_games_tokens
from stylometry.model import TokenTransformerEncoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Infer embeddings and write submission.csv')
    p.add_argument('--data-root', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--seq-len', type=int, default=64)
    p.add_argument('--arch', type=str, default='token', choices=['token','board'])
    return p.parse_args()


def _norm_id(stem: str) -> str:
    """Normalize file stem to numeric id string (e.g., 'player001' -> '1')."""
    m = re.search(r"\d+", stem)
    return str(int(m.group())) if m else stem


def embed_file(model: TokenTransformerEncoder, path: Path, device: torch.device, seq_len: int) -> np.ndarray:
    toks, msk = load_player_games_tokens(path, seq_len=seq_len)
    if toks.shape[0] == 0:
        return np.zeros((model.proj[-1].out_features,), dtype=np.float32)
    with torch.no_grad():
        t = torch.from_numpy(toks).to(device)
        m = torch.from_numpy(msk).to(device)
        emb = model(t, m)  # [G, D]
        emb = emb.mean(dim=0)  # [D]
        return emb.cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.arch == 'board':
        # Lazy import to avoid ImportError when only token model is needed
        from stylometry.model import BoardStylometryModel  # type: ignore
        from stylometry.boards import load_player_games_planes  # type: ignore
        model = BoardStylometryModel()
    else:
        model = TokenTransformerEncoder()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    root = Path(args.data_root)
    cand_dir = root / 'test_set' / 'cand_set'
    query_dir = root / 'test_set' / 'query_set'

    # Embed candidates
    cand_vecs: Dict[str, np.ndarray] = {}
    for p in sorted(cand_dir.glob('*.sgf'), key=lambda x: x.name):
        cid = _norm_id(p.stem)
        if args.arch == 'board':
            from stylometry.boards import load_player_games_planes  # lazy import
            planes, mask = load_player_games_planes(p, seq_len=args.seq_len)
            if planes.shape[0] == 0:
                vec = np.zeros((model.game_enc.proj[-1].out_features,), dtype=np.float32)
            else:
                with torch.no_grad():
                    B = planes.shape[0]
                    t_planes = torch.from_numpy(planes).to(device)  # [G,T,C,H,W]
                    t_mask = torch.from_numpy(mask).to(device)
                    vecs = model(t_planes, t_mask)  # [G,D]
                    vec = vecs.mean(dim=0).cpu().numpy().astype(np.float32)
            cand_vecs[cid] = vec
        else:
            cand_vecs[cid] = embed_file(model, p, device, args.seq_len)

    cand_ids = sorted(cand_vecs.keys(), key=lambda s: int(s) if s.isdigit() else s)
    C = np.stack([cand_vecs[c] for c in cand_ids], axis=0)  # [Nc, D]
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)

    rows: List[Tuple[str, str]] = []
    for p in sorted(query_dir.glob('*.sgf'), key=lambda x: x.name):
        qid = _norm_id(p.stem)
        if args.arch == 'board':
            from stylometry.boards import load_player_games_planes  # lazy import
            planes, mask = load_player_games_planes(p, seq_len=args.seq_len)
            if planes.shape[0] == 0:
                q = np.zeros((C.shape[1],), dtype=np.float32)
            else:
                with torch.no_grad():
                    t_planes = torch.from_numpy(planes).to(device)
                    t_mask = torch.from_numpy(mask).to(device)
                    vecs = model(t_planes, t_mask)
                    q = vecs.mean(dim=0).cpu().numpy().astype(np.float32)
        else:
            q = embed_file(model, p, device, args.seq_len)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = (C @ q)
        best = int(np.argmax(sims))
        rows.append((qid, cand_ids[best]))

    out_path = Path(args.out)
    with out_path.open('w', encoding='utf-8') as f:
        f.write('id,label\n')
        for qid, cid in rows:
            # Ensure IDs are numeric-only to match dataset submission format
            qid_num = _norm_id(qid)
            cid_num = _norm_id(cid)
            f.write(f'{qid_num},{cid_num}\n')
    print(f'Wrote {len(rows)} rows to {out_path}')


if __name__ == '__main__':
    main()
