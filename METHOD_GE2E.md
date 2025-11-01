## Go Stylometry GE2E Prototype

- Task: identify the player of Go SGF files (behavioral stylometry) using a token-based sequence encoder and GE2E loss.

### Data → Tokens
- Parse `*.sgf` (one file per player). Extract only the target player's moves per game using PB/PW tags.
- Map SGF coords to tokens in [0..360] (19×19), PASS=361. Truncate/pad each game's target-player moves to `seq_len` (default 64) with masks.

### Model
- Token embedding (362 → 128) + sinusoidal positions.
- 3× TransformerEncoder layers (d_model=128, heads=4, ff=256).
- Masked mean pooling → projection (128→128) → L2-normalize: per-game embedding.

### Loss (training)
- GE2E loss over batches of N players × M games: each game’s embedding is pushed toward its player centroid (leave-one-out) and away from other player centroids via cosine similarities with learnable (w, b).

### Inference
- For each candidate file: embed all games, average to candidate centroid.
- For each query file: embed all games, average; choose nearest candidate by cosine.

### Run (container)
Train (~<1h GPU, cached tokens on-the-fly):
```
python ML-Assignment2-Q5/train_ge2e.py \
  --data-root /workspace/machine-learning-class-fall-2025-assignment-2-q-5 \
  --save /workspace/stylometry_ckpt.pt --steps 20000 --seq-len 64 \
  --players-per-batch 32 --games-per-player 6 --amp 1
```

Infer + write submission:
```
python ML-Assignment2-Q5/infer_submit.py \
  --data-root /workspace/machine-learning-class-fall-2025-assignment-2-q-5 \
  --ckpt /workspace/stylometry_ckpt.pt --out /workspace/submission.csv
```

### Notes
- Speed knobs: reduce `--steps`, `--seq-len`, or layers in `stylometry/model.py` for faster runs.
- If PB/PW don’t match the first line player id, those games are skipped.
