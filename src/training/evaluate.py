from pathlib import Path
import glob
import yaml
from rich import print
from ultralytics import YOLO

from src.data.convert_coco_to_yolo import convert_coco_to_yolo
from src.training.train import _read_list, _write_dataset_yaml


def _find_latest_best(repo_root: Path) -> Path | None:
    # Try common Ultralytics run layout
    candidates = glob.glob(str(repo_root / "outputs" / "runs" / "**" / "weights" / "best.pt"), recursive=True)
    if not candidates:
        return None
    candidates.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(candidates[0])


def run_evaluation(model_path: Path, data_cfg: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_cfg = Path(data_cfg)
    assert data_cfg.exists(), f"Missing dataset config: {data_cfg}"

    with open(data_cfg, "r") as f:
        data = yaml.safe_load(f)

    annotations_path = repo_root / data.get("annotations", "annotations.json")
    images_dir = repo_root / data.get("val", data.get("train", "images"))
    splits_cfg = data.get("splits", {})
    val_list = repo_root / splits_cfg.get("val", "splits/val.txt")
    labels_dir = repo_root / "labels"

    assert annotations_path.exists(), f"Annotations JSON not found: {annotations_path}"
    assert images_dir.exists(), f"Images directory not found: {images_dir}"
    assert val_list.exists(), "Val split not found. Run: python -m src.cli split"

    val_images = _read_list(val_list)

    # Ensure labels and dataset yaml
    names = convert_coco_to_yolo(
        annotations_path=annotations_path,
        images_dir=images_dir,
        labels_dir=labels_dir,
        restrict_to_images=val_images,
    )
    tmp_yaml = repo_root / "outputs" / "tmp_dataset.yaml"
    dataset_yaml = _write_dataset_yaml(tmp_yaml, val_list, val_list, names)

    # Resolve model
    if model_path is None:
        model_path = _find_latest_best(repo_root)
        if model_path is None:
            raise FileNotFoundError("No trained model found. Please run training or pass --model-path")

    print(f"[bold cyan]Evaluating model:[/bold cyan] {model_path}")
    yolo = YOLO(str(model_path))
    metrics = yolo.val(data=str(dataset_yaml))
    print("[green]Evaluation complete.[/green]")
    print(metrics.results_dict)
