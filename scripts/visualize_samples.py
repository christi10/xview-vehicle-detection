from pathlib import Path
import typer
from src.data.check_dataset import check_dataset

app = typer.Typer(add_completion=False)


@app.command()
def main(
    annotations: Path = typer.Option(Path("annotations.json")),
    images_dir: Path = typer.Option(Path("images")),
    n: int = typer.Option(10, help="Number of samples to visualize"),
):
    check_dataset(annotations=annotations, images_dir=images_dir, sample_vis=n)


if __name__ == "__main__":
    app()
