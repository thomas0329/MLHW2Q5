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

```bash
python train_ge2e.py \
  --data-root /workspace/machine-learning-class-fall-2025-assignment-2-q-5 \
  --save /workspace/checkpoints/ge2e_token.pt \
  --seq-len 64 \
  --players-per-batch 32 \
  --games-per-player 6 \
  --steps 30000 \
  --lr 3e-4 \
  --logdir /workspace/runs/ge2e \
  --eval-interval 1000 \
  --eval-train-max-players 100 \
  --save-interval 5000
```

**Note:** Use backslash `\` (not `>`) for line continuation. If you encounter import errors after updating code, clear Python cache:
```bash
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
```

**New options:**
- `--eval-interval`: Evaluate accuracy every N steps (default: 1000). Set to 0 to disable evaluation.
- `--eval-train-max-players`: Limit number of players evaluated on training set for faster evaluation (default: 100).
- `--save-interval`: Save intermediate checkpoints every N steps (default: 5000). Set to 0 to disable periodic checkpoints.

**Note:** Checkpoints are saved in timestamped directories (e.g., `ge2e_token_20250115_143022/`) to separate different training runs. The final checkpoint is saved as the filename you specify within that directory.

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

**Available metrics:**
- `train/loss`: Training loss
- `train/grad_norm`: Gradient norm
- `train/w`, `train/b`: GE2E loss parameters
- `eval/train_acc`: Training accuracy (if `--eval-interval` > 0)
- `eval/test_acc`: Test accuracy (if `--eval-interval` > 0 and ground truth available)

### Inference (GE2E stylometry)
Given a trained checkpoint, write a `submission.csv` using cosine similarity search:

```bash
python infer_submit.py \
  --data-root /workspace/machine-learning-class-fall-2025-assignment-2-q-5 \
  --ckpt /workspace/checkpoints \
  --out /workspace/submission_stylometry.csv \
  --seq-len 64
```

**Note:** The `--ckpt` argument can be:
- A checkpoint file path (e.g., `/workspace/checkpoints/ge2e_token_20250115_143022/ge2e_token.pt`)
- A checkpoint directory (e.g., `/workspace/checkpoints/ge2e_token_20250115_143022/`) - will use the final checkpoint
- The parent checkpoint directory (e.g., `/workspace/checkpoints`) - will automatically find the latest checkpoint from the most recent training run

The script automatically finds the latest checkpoint when given a directory.

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



