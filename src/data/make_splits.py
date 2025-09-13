from pathlib import Path
from typing import List, Tuple
import random


def _scene_id_from_name(name: str) -> str:
    # Expect names like xView_<scene>_<tile>.tif
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[1]
    return stem  # fallback


def _collect_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".jpg", ".png"}])


def make_splits(images_dir: Path, train_ratio: float, seed: int, train_list: Path, val_list: Path) -> Tuple[List[Path], List[Path]]:
    images_dir = Path(images_dir)
    assert images_dir.exists(), f"Images directory not found: {images_dir}"

    all_imgs = _collect_images(images_dir)
    assert len(all_imgs) > 0, f"No images found in {images_dir}"

    # Group by scene id
    by_scene = {}
    for p in all_imgs:
        sid = _scene_id_from_name(p.name)
        by_scene.setdefault(sid, []).append(p)

    scenes = sorted(by_scene.keys())
    random.Random(seed).shuffle(scenes)

    train_cut = int(len(scenes) * train_ratio)
    train_scenes = set(scenes[:train_cut])

    train_imgs, val_imgs = [], []
    for sid, lst in by_scene.items():
        if sid in train_scenes:
            train_imgs.extend(lst)
        else:
            val_imgs.extend(lst)

    # Write absolute paths for Ultralytics compatibility with list files
    train_list.parent.mkdir(parents=True, exist_ok=True)
    val_list.parent.mkdir(parents=True, exist_ok=True)

    with open(train_list, "w") as f:
        for p in train_imgs:
            f.write(str(p.resolve()) + "\n")

    with open(val_list, "w") as f:
        for p in val_imgs:
            f.write(str(p.resolve()) + "\n")

    return train_imgs, val_imgs
