import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import time

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, total=None, desc=None):
        return it


def load_style_py(game_type: str, conf_path: Path):
    _temps = __import__(f'build.{game_type}', globals(), locals(), ['style_py'], 0)
    style_py = _temps.style_py
    style_py.load_config_file(str(conf_path.resolve()))
    return style_py


def ensure_cache_dir(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)


def hash_path(path: Path) -> str:
    return hashlib.md5(str(path.resolve()).encode('utf-8')).hexdigest()


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


def _fallback_simple_sgf_feature(sgf_text: str, dim: int = 256) -> np.ndarray:
    # Simple bag-of-characters hashing as a last resort feature
    v = np.zeros(dim, dtype=np.float32)
    for ch in sgf_text:
        v[ord(ch) % dim] += 1.0
    return v


def make_style_feature_fn(style_py) -> Callable[[Path], np.ndarray]:
    # Try a set of likely API names
    candidates: List[str] = [
        'extract_features_from_sgf',
        'extract_feature_from_sgf',
        'features_from_sgf',
        'featurize_sgf',
    ]

    extractor: Optional[Callable[[str], Iterable[float]]] = None
    for name in candidates:
        if hasattr(style_py, name):
            fn = getattr(style_py, name)
            if callable(fn):
                extractor = fn
                break

    def feature_fn(path: Path) -> np.ndarray:
        if extractor is not None:
            feats = extractor(str(path))
            arr = np.array(feats, dtype=np.float32)
            return arr
        # Fallback: best-effort simple feature
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            text = ''
        return _fallback_simple_sgf_feature(text)

    return feature_fn


def cached_feature(path: Path, cache_dir: Path, extractor: Callable[[Path], np.ndarray]) -> np.ndarray:
    key = hash_path(path)
    cache_path = cache_dir / f'{key}.npy'
    if cache_path.exists():
        return np.load(cache_path)
    vec = extractor(path)
    np.save(cache_path, vec)
    return vec


def load_sgf_files(dir_path: Path) -> List[Path]:
    return sorted([p for p in dir_path.glob('*.sgf') if p.is_file()], key=lambda p: p.name)


def resolve_test_subdir(test_root: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        candidate = test_root / name
        if candidate.exists() and candidate.is_dir():
            files = load_sgf_files(candidate)
            if len(files) > 0:
                return candidate
    return None


def build_prototypes(file_paths: List[Path], cache_dir: Path, extractor: Callable[[Path], np.ndarray], transform: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Dict[str, np.ndarray]:
    prototypes: Dict[str, np.ndarray] = {}
    start = time.time()
    for p in tqdm(file_paths, total=len(file_paths), desc='Building candidate features'):
        vec = cached_feature(p, cache_dir, extractor)
        if transform is not None:
            vec = transform(vec)
        player_id = p.stem  # filename without extension
        prototypes[player_id] = vec.astype(np.float32)
    elapsed = time.time() - start
    if len(file_paths) > 0:
        print(f'Candidate feature build: {len(file_paths)} files in {elapsed:.1f}s '
              f'(~{elapsed / len(file_paths):.3f}s/file)')
    return prototypes


def predict_queries(query_files: List[Path], candidate_proto: Dict[str, np.ndarray], cache_dir: Path, extractor: Callable[[
                    Path], np.ndarray], transform: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> List[Tuple[str, str]]:
    # Prepare candidate matrix
    cand_ids = sorted(candidate_proto.keys(), key=lambda x: int(x) if x.isdigit() else x)
    cand_mat = np.stack([candidate_proto[cid] for cid in cand_ids], axis=0)

    results: List[Tuple[str, str]] = []
    start = time.time()
    for q in tqdm(query_files, total=len(query_files), desc='Scoring queries'):
        q_id = q.stem
        q_vec = cached_feature(q, cache_dir, extractor)
        if transform is not None:
            q_vec = transform(q_vec)
        q_vec = q_vec[None, :]
        sims = cosine_similarity_matrix(q_vec.astype(np.float32), cand_mat)
        best_idx = int(np.argmax(sims[0]))
        pred_cid = cand_ids[best_idx]
        results.append((q_id, pred_cid))
    elapsed = time.time() - start
    if len(query_files) > 0:
        print(f'Query scoring: {len(query_files)} files in {elapsed:.1f}s '
              f'(~{elapsed / len(query_files):.3f}s/file)')
    return results


def compute_train_normalizer(train_files: List[Path], cache_dir: Path, extractor: Callable[[Path], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    vecs: List[np.ndarray] = []
    for p in tqdm(train_files, total=len(train_files), desc='Building train features (for normalization)'):
        vec = cached_feature(p, cache_dir, extractor)
        vecs.append(vec.astype(np.float32))
    if not vecs:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    X = np.stack(vecs, axis=0)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def write_submission(rows: List[Tuple[str, str]], out_path: Path) -> None:
    with out_path.open('w', encoding='utf-8') as f:
        f.write('id,label\n')
        for qid, cid in rows:
            f.write(f'{qid},{cid}\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Q5 Go Player Identification using style_py')
    parser.add_argument('--data-root', type=str, required=True, help='Root folder containing train_set/ and test_set/')
    parser.add_argument('--conf', type=str, default='conf.cfg', help='Path to conf.cfg for style_py')
    parser.add_argument('--out', type=str, default='submission.csv', help='Output submission.csv path')
    parser.add_argument('--cache-dir', type=str, default='.cache', help='Cache directory for features')
    parser.add_argument('--game-type', type=str, default='go', help='Game type for import (default: go)')
    parser.add_argument('--norm', type=str, default='train_zscore', choices=['none', 'train_zscore'], help='Feature normalization strategy')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    conf_path = Path(args.conf)
    out_path = Path(args.out)
    cache_dir = Path(args.cache_dir)

    # style_py import and config
    style_py = load_style_py(args.game_type, conf_path)
    extractor = make_style_feature_fn(style_py)
    ensure_cache_dir(cache_dir)

    # Paths
    train_dir = data_root / 'train_set'
    test_root = data_root / 'test_set'

    # Robustly resolve query/candidate directories (support multiple common names)
    query_dir = resolve_test_subdir(test_root, ['query_set', 'query', 'queries']) or (test_root / 'query')
    cand_dir = resolve_test_subdir(test_root, ['cand_set', 'candidate', 'candidates']) or (test_root / 'candidate')

    # Load file lists
    query_files = load_sgf_files(query_dir)
    cand_files = load_sgf_files(cand_dir)

    # Early validations with actionable messages
    if len(cand_files) == 0:
        raise RuntimeError(
            f'No candidate SGF files found. Checked: {cand_dir}. '
            f'Ensure SGFs exist under test_set/cand_set, test_set/candidate, or test_set/candidates. '
            f'Data root: {data_root}'
        )
    if len(query_files) == 0:
        raise RuntimeError(
            f'No query SGF files found. Checked: {query_dir}. '
            f'Ensure SGFs exist under test_set/query_set, test_set/query, or test_set/queries. '
            f'Data root: {data_root}'
        )

    print(f'Found {len(cand_files)} candidate files in {cand_dir}')
    print(f'Found {len(query_files)} query files in {query_dir}')

    # Optional normalization using train_set statistics
    transform_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    if args.norm == 'train_zscore':
        train_files = load_sgf_files(train_dir)
        if len(train_files) > 0:
            mean, std = compute_train_normalizer(train_files, cache_dir, extractor)
            if mean.size > 0:
                print('Applying train-set z-score normalization')

                def transform_fn(v: np.ndarray) -> np.ndarray:
                    return (v.astype(np.float32) - mean) / std
        else:
            print('No train files found for normalization; proceeding without normalization')

    # Build candidate prototypes (one file -> one prototype)
    candidate_prototypes = build_prototypes(cand_files, cache_dir, extractor, transform=transform_fn)

    # Predict each query by cosine to candidate prototypes
    overall_start = time.time()
    preds = predict_queries(query_files, candidate_prototypes, cache_dir, extractor, transform=transform_fn)
    overall_elapsed = time.time() - overall_start

    # Write submission
    write_submission(preds, out_path)

    print(f'Wrote submission to {out_path.resolve()} (rows={len(preds)})')
    print(f'Total prediction stage elapsed: {overall_elapsed:.1f}s')


if __name__ == '__main__':
    main()
