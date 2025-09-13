# Implementation Plan

This document outlines the repository structure, design patterns, toolchain choices, and the step-by-step plan to implement training, evaluation, and inference for the xView COCO-style dataset contained in this repo.

## Tech Stack Decision

- Framework: PyTorch + Ultralytics (YOLO)
- Why not TensorFlow: TensorFlow installs on macOS as `tensorflow-macos` + `tensorflow-metal` (GPU via Metal) but compatibility can vary and wheel sizes are larger. PyTorch CPU wheels install reliably via pip on macOS and Ultralytics offers an end-to-end COCO-friendly pipeline that reduces boilerplate.
- Benefits:
  - Simple pip installation (no custom builds required on macOS CPU)
  - Native COCO support out of the box
  - Strong docs and fast iteration for training and inference

### Quick Install (macOS, CPU)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python matplotlib pycocotools tqdm typer rich pyyaml
```

Notes:
- If you have a recent Apple Silicon Mac and want TensorFlow instead, you could do:
  ```bash
  pip install tensorflow-macos tensorflow-metal
  ```
  but this plan proceeds with PyTorch + Ultralytics for simplicity.

## Repository Structure (to create)

We will add the following structure on top of the existing dataset files:

```
assesment2/
├── images/                      # Provided training/validation image tiles (640x640 tif)
├── future_pass_images/          # Provided future inference tiles
├── annotations.json             # COCO-format annotations (images/annotations/categories)
├── Problem description.pdf      # Project brief
├── IMPLEMENTATION_PLAN.md       # This document
├── configs/
│   ├── dataset.yaml             # Ultralytics dataset config (paths, class names)
│   ├── train.yaml               # Training hyperparameters and model choice
│   └── eval.yaml                # Evaluation/config overrides
├── splits/
│   ├── train.txt                # List of image filenames for train split
│   └── val.txt                  # List of image filenames for val split
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── check_dataset.py     # Consistency checks, stats, and quick visualizations
│   │   └── make_splits.py       # Deterministic split creation (scene-aware or stratified)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py             # Launch Ultralytics training using configs
│   │   └── evaluate.py          # Run COCO-style eval; produce metrics artifacts
│   ├── inference/
│   │   ├── __init__.py
│   │   └── infer.py             # Batch inference for future_pass_images/ with outputs
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── io.py                # I/O helpers (paths, JSON read/write)
│   │   ├── vis.py               # Visualization helpers (bboxes on tiles)
│   │   └── metrics.py           # Optional metric helpers beyond YOLO's built-ins
│   └── cli.py                   # Unified Typer-based CLI entrypoint
├── scripts/
│   ├── visualize_samples.py     # Save random annotated image previews
│   └── export_results.py        # Convert predictions to CSV/COCO formats
├── outputs/
│   ├── runs/                    # Training runs/checkpoints (Ultralytics default)
│   ├── eval/                    # Evaluation reports (mAP, PR curves, confusions)
│   └── vis/                     # Visualized GT and predictions
├── requirements.txt             # Frozen dependencies for reproducibility
└── README.md                    # Usage instructions
```

We will create minimal versions of these files first, then iterate.

## Design Patterns & Conventions

- Modular architecture:
  - Separate concerns into `data/`, `training/`, `inference/`, and `utils/` packages.
- Configuration-driven:
  - YAML files under `configs/` define dataset paths, model architecture, and hyperparams.
  - Scripts read configs rather than hardcoding settings.
- Strategy pattern for transforms:
  - Define augmentation strategies in config; Ultralytics handles most common augmentations. Custom strategies can be toggled.
- Factory pattern for runners:
  - A simple factory chooses model variant (e.g., YOLOn, custom backbone) based on config.
- Dependency inversion for I/O:
  - `utils/io.py` abstracts reading/writing JSON, CSV, COCO results to keep upper layers clean.
- Reproducibility & observability:
  - Seeded runs; logs and artifacts go to `outputs/` with run IDs.
  - Use `rich` for readable CLI logs; save metrics and plots.
- CLI-first workflow:
  - `src/cli.py` exposes subcommands: `check`, `split`, `train`, `eval`, `infer`.
  - Encourages consistent usage and easy automation.

## Planned Workflow

1. Dataset checks and stats
   - `src/data/check_dataset.py`: verify that every `images/` file has a matching `images` entry in `annotations.json`, ensure all `annotations.image_id` refer to valid images, compute per-category counts, and optionally save a few annotated previews to `outputs/vis/`.

2. Create train/val splits
   - `src/data/make_splits.py`: generate `splits/train.txt` and `splits/val.txt` deterministically. If scenes are encoded in names (e.g., `xView_<sceneId>_<tile>`), keep tiles from the same scene together to reduce leakage.

3. Configure Ultralytics dataset YAML
   - `configs/dataset.yaml`: point to `images/` for train/val; list class names from COCO `categories`. For Ultralytics, if using COCO JSON, we can also rely on native COCO support or convert if helpful.

4. Training
   - `src/training/train.py`: load `train.yaml` with chosen model (e.g., `yolov8n`/`yolo11n`), epochs, imgsz, batch size, data config. Start/monitor training, save runs under `outputs/runs/`.

5. Evaluation
   - `src/training/evaluate.py`: run validation on the val split; export metrics (mAP@[.5:.95], per-class AP), confusion matrices, and PR curves to `outputs/eval/`.

6. Inference on future tiles
   - `src/inference/infer.py`: run batch inference over `future_pass_images/`, save:
     - Visual overlays in `outputs/vis/`
     - COCO-format detections JSON
     - CSV with `image, x_min, y_min, x_max, y_max, score, category`

7. Packaging & Docs
   - `README.md`: environment setup, commands for `check`, `split`, `train`, `eval`, `infer`.
   - `requirements.txt`: pinned versions once verified locally.

## Initial Milestones

- M1: Create scaffolding files/folders and CLI skeleton
- M2: Implement dataset checks and splits; commit `configs/dataset.yaml`
- M3: Baseline training run (nano/small model) to confirm pipeline
- M4: Evaluation artifacts and error analysis
- M5: Inference over `future_pass_images/` and export deliverables

## Open Questions (to refine after confirming PDF details)

- Exact category list and any merging/remapping?
- Required evaluation metrics/thresholds? Any specific AP target?
- Any constraints on runtime (CPU-only) or training budget (epochs/time)?

## Next Steps

- I will scaffold `configs/`, `src/`, `scripts/`, and `outputs/` with minimal placeholders and prepare `requirements.txt` and `README.md` in subsequent commits so we can start running checks immediately.
