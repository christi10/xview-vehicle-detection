from pathlib import Path
import yaml
from rich import print
from ultralytics import YOLO

from src.data.convert_coco_to_yolo import convert_coco_to_yolo


def _read_list(list_path: Path) -> list[Path]:
    with open(list_path, "r") as f:
        return [Path(line.strip()) for line in f if line.strip()]


def _write_dataset_yaml(tmp_yaml: Path, train_list: Path, val_list: Path, names: list[str]) -> Path:
    tmp_yaml.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": str(train_list.resolve()),
        "val": str(val_list.resolve()),
        "names": names,
        "nc": len(names),
    }
    tmp_yaml.write_text(yaml.safe_dump(payload))
    return tmp_yaml


def run_training(train_cfg: Path, data_cfg: Path) -> None:
    """Train a YOLO model using Ultralytics with split lists and COCO annotations.

    Steps:
    - Load configs
    - Read splits/train.txt and splits/val.txt
    - Convert COCO annotations to YOLO label files for images in train+val
    - Generate a dataset YAML pointing to the list files and class names
    - Launch YOLO training
    """
    train_cfg = Path(train_cfg)
    data_cfg = Path(data_cfg)

    assert train_cfg.exists(), f"Missing training config: {train_cfg}"
    assert data_cfg.exists(), f"Missing dataset config: {data_cfg}"

    with open(train_cfg, "r") as f:
        train = yaml.safe_load(f)
    with open(data_cfg, "r") as f:
        data = yaml.safe_load(f)

    # Paths
    repo_root = Path(__file__).resolve().parents[2]
    annotations_path = repo_root / data.get("annotations", "annotations.json")
    images_dir = repo_root / data.get("train", "images")
    splits_cfg = data.get("splits", {})
    train_list = repo_root / splits_cfg.get("train", "splits/train.txt")
    val_list = repo_root / splits_cfg.get("val", "splits/val.txt")
    labels_dir = repo_root / "labels"

    assert annotations_path.exists(), f"Annotations JSON not found: {annotations_path}"
    assert images_dir.exists(), f"Images directory not found: {images_dir}"
    assert train_list.exists() and val_list.exists(), "Split lists not found. Run: python -m src.cli split"

    train_images = _read_list(train_list)
    val_images = _read_list(val_list)

    print("[bold cyan]== Converting COCO -> YOLO labels ==[/bold cyan]")
    names = convert_coco_to_yolo(
        annotations_path=annotations_path,
        images_dir=images_dir,
        labels_dir=labels_dir,
        restrict_to_images=train_images + val_images,
    )

    # Create temporary dataset yaml pointing to list files
    tmp_yaml = repo_root / "outputs" / "tmp_dataset.yaml"
    dataset_yaml = _write_dataset_yaml(tmp_yaml, train_list, val_list, names)

    # Model selection
    model_name = train.get("model", "yolov8n")
    imgsz = int(train.get("imgsz", 640))
    epochs = int(train.get("epochs", 50))
    batch = int(train.get("batch", 16))
    seed = int(train.get("seed", 42))
    patience = int(train.get("patience", 20))
    lr0 = float(train.get("lr0", 0.01))
    weight_decay = float(train.get("weight_decay", 0.0005))

    print("[bold cyan]== Launching Training ==[/bold cyan]")
    print({
        "model": model_name,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "seed": seed,
        "patience": patience,
        "lr0": lr0,
        "weight_decay": weight_decay,
    })

    yolo = YOLO(model_name)
    yolo.train(
        data=str(dataset_yaml),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        seed=seed,
        patience=patience,
        lr0=lr0,
        weight_decay=weight_decay,
        project=str((repo_root / "outputs" / "runs").resolve()),
        name="train",
        verbose=True,
    )
    print("[green]Training run complete. Check outputs/runs for logs and weights.[/green]")
