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

### Training (GE2E stylometry)
From repo root inside the container:

```
python train_ge2e.py \
>   --data-root /workspace/machine-learning-class-fall-2025-assignment-2-q-5 \
>   --save /workspace/checkpoints/ge2e_token.pt \
>   --seq-len 64 \
>   --players-per-batch 32 \
>   --games-per-player 6 \
>   --steps 30000 \
>   --lr 3e-4 \
>   --logdir /workspace/runs/ge2e
```

### Visualization
To view training curves with TensorBoard:

**From inside the container:**
```bash
tensorboard --logdir /workspace/runs/ge2e --host 0.0.0.0 --port 6006
```

**From host machine:**
```bash
tensorboard --logdir /home/thomas/Desktop/ml_hw2/ML-Assignment2-Q5/runs/ge2e --port 6006
```

Then open `http://localhost:6006` in your browser.

### Inference (GE2E stylometry)
Given a trained checkpoint, write a `submission.csv` using cosine similarity search:

```
python infer_submit.py \
  --data-root /workspace/machine-learning-class-fall-2025-assignment-2-q-5 \
  --ckpt /workspace/checkpoints/ge2e_token.pt \
  --out /workspace/submission_stylometry.csv \
  --seq-len 64
```

### Inference (style_py baseline)
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



