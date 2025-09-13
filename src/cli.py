import json
import os
from pathlib import Path
import typer
from rich import print

from src.data.check_dataset import check_dataset
from src.data.make_splits import make_splits
from src.training.train import run_training
from src.training.evaluate import run_evaluation
from src.inference.infer import run_inference

app = typer.Typer(add_completion=False, help="CLI for dataset checks, splits, training, evaluation, and inference.")

REPO_ROOT = Path(__file__).resolve().parents[1]


@app.command()
def check(
    annotations: Path = typer.Option(REPO_ROOT / "annotations.json", exists=True, readable=True),
    images_dir: Path = typer.Option(REPO_ROOT / "images", exists=True),
    sample_vis: int = typer.Option(0, help="Number of random samples to visualize to outputs/vis/ (0=skip)"),
):
    """Validate COCO JSON against images/ and print basic stats."""
    ok = check_dataset(annotations=annotations, images_dir=images_dir, sample_vis=sample_vis)
    raise typer.Exit(code=0 if ok else 1)


@app.command()
def split(
    train_ratio: float = typer.Option(0.9, min=0.5, max=0.99),
    seed: int = typer.Option(42),
    images_dir: Path = typer.Option(REPO_ROOT / "images", exists=True),
    train_list: Path = typer.Option(REPO_ROOT / "splits" / "train.txt"),
    val_list: Path = typer.Option(REPO_ROOT / "splits" / "val.txt"),
):
    """Create deterministic train/val splits, grouping by scene id to avoid leakage."""
    make_splits(images_dir=images_dir, train_ratio=train_ratio, seed=seed, train_list=train_list, val_list=val_list)
    print(f"[green]Wrote splits to[/green] {train_list} and {val_list}")


@app.command()
def train(
    config: Path = typer.Option(REPO_ROOT / "configs" / "train.yaml"),
    dataset_config: Path = typer.Option(REPO_ROOT / "configs" / "dataset.yaml"),
):
    """Run Ultralytics training according to configs."""
    run_training(train_cfg=config, data_cfg=dataset_config)


@app.command()
def eval(
    model_path: Path = typer.Option(None, help="Path to trained model (e.g., outputs/runs/detect/train/weights/best.pt). If None, use latest."),
    dataset_config: Path = typer.Option(REPO_ROOT / "configs" / "dataset.yaml"),
):
    """Evaluate a trained model on the val split."""
    run_evaluation(model_path=model_path, data_cfg=dataset_config)


@app.command()
def infer(
    model_path: Path = typer.Option(None, help="Path to trained model .pt. If None, use latest best."),
    images_dir: Path = typer.Option(REPO_ROOT / "future_pass_images", exists=True),
    conf: float = typer.Option(0.25),
    iou: float = typer.Option(0.45),
):
    """Run inference on future_pass_images/ and save visualizations and predictions."""
    run_inference(model_path=model_path, images_dir=images_dir, conf=conf, iou=iou)


if __name__ == "__main__":
    app()
