import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def _build_category_mapping(categories: List[Dict[str, Any]]) -> Tuple[Dict[int, int], List[str]]:
    """Map original COCO category ids to contiguous [0..N-1] indices.
    Returns (id_map, names_list)
    """
    names = [c["name"] for c in categories]
    # Preserve order in categories list
    id_map = {c["id"]: i for i, c in enumerate(categories)}
    return id_map, names


def _xywh_to_yolo(bbox, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    # COCO bbox is [x_min, y_min, width, height]
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    wn = w / img_w
    hn = h / img_h
    return xc, yc, wn, hn


def convert_coco_to_yolo(
    annotations_path: Path,
    images_dir: Path,
    labels_dir: Path,
    restrict_to_images: List[Path] | None = None,
) -> List[str]:
    """Convert COCO detection annotations to YOLO .txt label files.

    - annotations_path: path to annotations.json
    - images_dir: directory containing the images referenced by file_name
    - labels_dir: output directory to place .txt files (mirrors image filenames)
    - restrict_to_images: optional list of image Paths to limit conversion (train/val split)

    Returns the ordered list of class names.
    """
    annotations_path = Path(annotations_path)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(annotations_path, "r") as f:
        coco: Dict[str, Any] = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    id_map, names = _build_category_mapping(cats)

    # Index images by both id and file_name
    image_by_id = {im["id"]: im for im in images}
    id_by_filename = {im["file_name"]: im["id"] for im in images}

    # Build per-image annotation list
    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for a in anns:
        iid = a.get("image_id")
        if iid is None or iid not in image_by_id:
            continue
        anns_by_img.setdefault(iid, []).append(a)

    # Helper for whether to include a given filename
    restrict_set = None
    if restrict_to_images is not None:
        restrict_set = {Path(p).name for p in restrict_to_images}

    # For each image entry, write labels
    for im in images:
        fname = im["file_name"]
        if restrict_set is not None and fname not in restrict_set:
            # skip files not in the requested subset
            continue
        img_w, img_h = int(im["width"]), int(im["height"])
        label_path = labels_dir / (Path(fname).with_suffix(".txt").name)
        lines: List[str] = []
        for a in anns_by_img.get(im["id"], []):
            if "bbox" not in a or a.get("iscrowd", 0) == 1:
                continue
            cid = a.get("category_id")
            if cid not in id_map:
                continue
            cls = id_map[cid]
            xc, yc, wn, hn = _xywh_to_yolo(a["bbox"], img_w, img_h)
            # Clamp to [0,1] just in case
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            wn = min(max(wn, 0.0), 1.0)
            hn = min(max(hn, 0.0), 1.0)
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
        label_path.write_text("\n".join(lines))

    return names
