## Q5 Method: Go Player Identification via Prototype Retrieval

### Overview
We frame player identification as retrieval: compute a fixed-length feature for each SGF file, build a prototype per candidate player, and predict the candidate with maximum cosine similarity to the query feature. We use the provided C++ backend via pybind (`style_py`) for feature extraction when available; otherwise, a minimal hashed-text fallback keeps the pipeline functional (lower accuracy).

### Data Layout
- Train: `machine-learning-class-fall-2025-assignment-2-q-5/train_set/*.sgf`
- Test:
  - Query: `.../test_set/{query_set|query|queries}/*.sgf`
  - Candidate: `.../test_set/{cand_set|candidate|candidates}/*.sgf`

### Pipeline
1) Import env and config
   - Import `build.go.style_py` and load `conf.cfg` as specified.

2) Feature extraction
   - Autodetects a `style_py` extractor (e.g., `extract_features_from_sgf`).
   - If none found, uses a simple hashed-character feature as a fallback.
   - All features are cached under `--cache-dir` to avoid recomputation.

3) Optional normalization (default: enabled)
   - Compute z-score statistics (mean/std) over features from `train_set`.
   - Apply `(x-mean)/std` to both candidate and query features prior to scoring.

4) Candidate prototypes
   - One SGF per candidate file → one feature vector → prototype.
   - Store in a matrix for efficient batched cosine scoring.

5) Retrieval and prediction
   - For each query, compute cosine similarity against all candidate prototypes.
   - Assign the candidate ID with the highest similarity.
   - Write `submission.csv` with `id,label` rows for all queries.

### Implementation Notes
- Entry point: `ML-Assignment2-Q5/Q5.py`
- CLI flags:
  - `--data-root` path to dataset root
  - `--conf` path to `conf.cfg`
  - `--cache-dir` feature cache directory
  - `--norm {train_zscore,none}` normalization strategy
- Progress: `tqdm` progress bars for building features and scoring queries.

### Runtime & Complexity
- Let Q be number of query files and C be number of candidate files.
  - Feature pass: O(Q + C) extractions (cached after first run).
  - Scoring: O(Q × C × d) where d is feature dimension; implemented as dense matrix ops.
- On an RTX 4500 class machine (CPU path), scoring 600×600 with cached features takes ~30–60s; the first uncached run depends on the extractor cost.

### Expected Accuracy
- With native `style_py` features: a simple prototype + cosine baseline typically achieves mid-range accuracy (estimate: 40–70% top-1). Train-set z-score may add 2–5%.
- With fallback hashed-text features: accuracy is near random (≈1/600 ≈ 0.17%).

### Reproduction
Inside the container from repo root:
```
python Q5.py \
  --data-root /workspace/machine-learning-class-fall-2025-assignment-2-q-5 \
  --conf /workspace/conf.cfg \
  --out /workspace/submission.csv \
  --cache-dir /workspace/.cache \
  --norm train_zscore
```

### Limitations & Future Work
- Dependence on `style_py` API autodetection; wiring the exact extractor improves robustness.
- Prototype averaging ignores intra-player variability; multi-prototype clustering or learned metric could help.
- Cosine on global features may underweight sequence dynamics; consider sequence models or per-move statistics.
- Calibration and post-processing (e.g., temperature scaling) could reduce ties and improve ranking quality.


