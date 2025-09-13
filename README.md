# xView Vehicle Detection — Training/Evaluation/Inference

This repository implements a complete pipeline to detect and count vehicles on xView 640×640 tiles. It includes:

- A Jupyter notebook to run the full workflow end-to-end (train/eval/infer and generate `results.json`).
- A CLI for dataset checks, splitting, training, evaluation, and inference.
- Config-driven setup to make experiments reproducible and easy to tweak.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional (register the venv as a Jupyter kernel):

```bash
python -m ipykernel install --user --name assesment2-nb --display-name "Python (assesment2)"
```

## Notebook-first workflow

Open the main notebook and run all cells:

```bash
jupyter notebook notebooks/solution.ipynb
```

The notebook will:

- Load the latest trained weights from `outputs/runs/**/weights/best.pt` (or you can train first with the CLI below).
- Evaluate on the validation split and log metrics (precision, recall, mAP50, mAP50–95).
- Run inference on `future_pass_images/` and count `small-vehicle` and `large-vehicle` detections per image.
- Write the deliverable JSON to `outputs/results.json` using the schema in this README.

> Tip: If no weights exist yet, use the CLI to create splits and run a quick baseline training (see next section).

### Notebook internals and developer usage

The notebook `notebooks/solution.ipynb` is structured as follows:

- "Environment and imports": resolves `REPO_ROOT` so relative paths work whether you run from `notebooks/` or repo root.
- "Configuration": reads `configs/train.yaml` and `configs/dataset.yaml` for consistency with the CLI.
- "Load Trained Model": finds the latest `best.pt` under `outputs/runs/**/weights/best.pt` and loads it with Ultralytics `YOLO`.
- "Evaluation on Validation Split": builds a minimal dataset YAML that points to `splits/val.txt` and calls `model.val()`; collects precision, recall, mAP50, mAP50–95.
- "Inference on `future_pass_images/` and Vehicle Counting": runs `model.predict()` on the folder and counts detections for `small-vehicle` and `large-vehicle` by class name.
- "Build `results.json`": writes `outputs/results.json` in the required submission schema.

Common customizations for developers:

- Change inference folder to test new data:
  - Edit the cell that defines `future_dir` and set `future_dir = Path('/absolute/path/to/your/images')`.
- Adjust detection thresholds:
  - In the predict cell, change `conf=0.25` and `iou=0.45` to your desired values (e.g., lower `conf` for higher recall).
- Re-train with different hyperparameters:
  - Edit `configs/train.yaml` (e.g., set `imgsz: 640`, `epochs: 50`, `model: yolov8s`) and re-run training via CLI: `python -m src.cli train`.
- Use different weights in the notebook:
  - Skip the auto-discovery and set `model_path = Path('outputs/runs/<your_run>/weights/best.pt')` before `YOLO(model_path)`.
- Export additional artifacts:
  - Visual predictions are written under `outputs/vis/infer_nb/`. You can also save the detections table to CSV by adding a small snippet that writes `results_list` to disk if needed.

## CLI commands (optional but recommended for training)

All commands are exposed via a unified CLI.

```bash
# Show available subcommands
python -m src.cli --help

# 1) Dataset checks and stats
python -m src.cli check

# 2) Create splits (deterministic; scene-aware)
python -m src.cli split --train-ratio 0.9

# 3) Train baseline (Ultralytics)
# Training hyperparameters are set in configs/train.yaml (default: yolov8n, imgsz=512, epochs=5 for quick runs)
python -m src.cli train

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

## Results JSON schema

The submission requires a JSON file with per-image counts for `small-vehicle` and `large-vehicle`, plus overall metrics from training/evaluation.

Example structure (what the notebook writes to `outputs/results.json`):

```json
{
  "results": [
    {
      "image_id": "xView_001_0.tif",
      "small_vehicle_count": 15,
      "large_vehicle_count": 5
    }
    // ... one entry per image in future_pass_images/
  ],
  "overall_metrics": {
    "train": {
      "average_precision": 0.78,
      "average_recall": 0.83,
      "average_mAP50": 0.77,
      "average_mAP50_95": 0.23
    }
  }
}
```

Notes:

- The notebook maps class ids to names and counts detections whose name matches small/large vehicle variants (`small-vehicle`, `small_vehicle`, etc.).
- Ultralytics exposes aggregate metrics via `model.val()`. IoU is not provided as a single averaged value; we include precision/recall/mAP50/mAP50–95.

## Configuration

- `configs/train.yaml` — Training hyperparameters and model choice. Defaults are set for quick iteration:
  - `model: yolov8n`
  - `imgsz: 512`
  - `epochs: 5`
  - Adjust for stronger results (e.g., `imgsz: 640`, `epochs: 50`, `model: yolov8s`).
- `configs/dataset.yaml` — Dataset and split configuration. Uses list files in `splits/` and `annotations.json`.

## Assumptions

- COCO `annotations.json` includes categories with names for vehicles such as `small-vehicle` and `large-vehicle`.
- Tiles are 640×640 `.tif` images (as provided). Ultralytics supports TIFF via OpenCV.
- CPU training on macOS is acceptable for quick baselines; longer runs improve accuracy.

## Reproducibility & tips

- Set seeds in `configs/train.yaml` (`seed: 42`).
- For faster debugging: reduce `epochs`, use `imgsz: 512`, keep `yolov8n`.
- For better accuracy: increase `epochs`, `imgsz`, and/or use `yolov8s` or larger, if resources allow.


## Troubleshooting

- If no weights are found when running the notebook, run `python -m src.cli split` and `python -m src.cli train` first.
- If Jupyter does not show the venv kernel, (re)install: `python -m ipykernel install --user --name assesment2-nb`.
- If you get OpenCV TIFF issues, ensure `opencv-python` is installed (already in `requirements.txt`).
