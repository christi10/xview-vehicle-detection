from pathlib import Path
import glob
import csv
from rich import print
from ultralytics import YOLO


def _find_latest_best(repo_root: Path) -> Path | None:
    candidates = glob.glob(str(repo_root / "outputs" / "runs" / "**" / "weights" / "best.pt"), recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(candidates[0])


def run_inference(model_path: Path, images_dir: Path, conf: float, iou: float) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    images_dir = Path(images_dir)
    assert images_dir.exists(), f"Images directory not found: {images_dir}"

    if model_path is None:
        model_path = _find_latest_best(repo_root)
        if model_path is None:
            raise FileNotFoundError("No trained model found. Train a model or pass --model-path")

    out_dir = repo_root / "outputs"
    vis_dir = out_dir / "vis"
    pred_dir = out_dir / "predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"[bold cyan]Running inference with:[/bold cyan] {model_path}")
    yolo = YOLO(str(model_path))
    results = yolo.predict(
        source=str(images_dir),
        conf=conf,
        iou=iou,
        save=True,
        project=str(vis_dir),
        name="infer",
        save_txt=False,
        verbose=True,
    )

    # Export a CSV with detections across all images
    csv_path = pred_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x_min", "y_min", "x_max", "y_max", "score", "class_id", "class_name"])
        for r in results:
            p = Path(r.path)
            names = r.names
            if r.boxes is None:
                continue
            for box in r.boxes:
                # xyxy, conf, cls
                xyxy = box.xyxy[0].tolist()
                x_min, y_min, x_max, y_max = [float(v) for v in xyxy]
                score = float(box.conf[0].item()) if box.conf is not None else 0.0
                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                writer.writerow([p.name, x_min, y_min, x_max, y_max, score, cls_id, cls_name])

    print(f"[green]Inference complete. Visuals saved under {vis_dir}/infer, CSV at {csv_path}[/green]")
