# xView Tiles — Training/Evaluation/Inference

This project trains an object detector on xView 640x640 tiles provided in `images/` with COCO-style `annotations.json`, and runs inference on `future_pass_images/`.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Commands

All commands are exposed via a unified CLI.

```bash
# Show available subcommands
python -m src.cli --help

# 1) Dataset checks and stats
python -m src.cli check

# 2) Create splits (deterministic; scene-aware)
python -m src.cli split --train-ratio 0.9

# 3) Train baseline (Ultralytics)
python -m src.cli train --epochs 50 --imgsz 640 --model yolov8n

# 4) Evaluate
python -m src.cli eval

# 5) Inference on future tiles
python -m src.cli infer --conf 0.25
```

## Layout

- `images/`: training/validation tiles
- `future_pass_images/`: future inference tiles
- `annotations.json`: COCO annotations
- `configs/`: dataset and training configs
- `splits/`: lists of filenames for train/val
- `src/`: project source code (data, training, inference, utils)
- `outputs/`: run artifacts, evaluation, and visualizations

See `IMPLEMENTATION_PLAN.md` for the full plan and design patterns.
