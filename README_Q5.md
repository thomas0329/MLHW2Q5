## Q5: Go Player Identification (using style_py)

### Prerequisites
- Build inside the provided container and compile Go env:
  - `./scripts/start-container.sh`
  - `./scripts/build.sh go`
  - Ensure `conf.cfg` exists (default `/workspace/conf.cfg`).

### Data
Place data under:
- Train: `ML-Assignment2-Q5/machine-learning-class-fall-2025-assignment-2-q-5/train_set/`
- Test:
  - Query: `ML-Assignment2-Q5/machine-learning-class-fall-2025-assignment-2-q-5/test_set/query/`
  - Candidate: `ML-Assignment2-Q5/machine-learning-class-fall-2025-assignment-2-q-5/test_set/candidate/`

### Run
From repo root inside the container (so `build/go` is importable):

```
python ML-Assignment2-Q5/Q5.py \
  --data-root /home/thomas/Desktop/ml_hw2/ML-Assignment2-Q5/machine-learning-class-fall-2025-assignment-2-q-5 \
  --conf /workspace/conf.cfg \
  --out /workspace/submission.csv \
  --cache-dir /workspace/.cache
```

### What it does
- Imports `style_py`, loads config, extracts a fixed-length vector per SGF file
- Builds a prototype per candidate (one vector per candidate file)
- Scores each query against all candidates via cosine similarity
- Writes `submission.csv` matching the sample format

### Notes
- You can override config path with `--conf`.
- Cached vectors are stored under `--cache-dir`.
- If `style_py` lacks a direct SGF extractor, the script falls back to a simple hashed character feature to remain functional.



