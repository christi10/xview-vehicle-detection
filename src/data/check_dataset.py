import json
from pathlib import Path
from collections import Counter, defaultdict
import random
from typing import Dict, Any

import matplotlib.pyplot as plt
import cv2


def check_dataset(annotations: Path, images_dir: Path, sample_vis: int = 0) -> bool:
    annotations = Path(annotations)
    images_dir = Path(images_dir)

    assert annotations.exists(), f"Annotations file not found: {annotations}"
    assert images_dir.exists(), f"Images directory not found: {images_dir}"

    with open(annotations, "r") as f:
        coco: Dict[str, Any] = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    # Indexes
    image_by_id = {im["id"]: im for im in images}
    cat_by_id = {c["id"]: c for c in cats}

    # Check that all image files exist
    missing_files = []
    for im in images:
        fp = images_dir / im["file_name"]
        if not fp.exists():
            missing_files.append(str(fp))

    # Check that all annotations reference valid image and category ids
    bad_anns = []
    cat_counter = Counter()
    img_ann_counter = Counter()
    for a in anns:
        iid = a.get("image_id")
        cid = a.get("category_id")
        if iid not in image_by_id or cid not in cat_by_id:
            bad_anns.append(a.get("id", None))
            continue
        cat_counter[cid] += 1
        img_ann_counter[iid] += 1

    # Summary
    print("== Dataset Summary ==")
    print(f"Images in JSON: {len(images)}")
    print(f"Annotations in JSON: {len(anns)}")
    print(f"Categories: {len(cats)}")

    if cats:
        print("Top categories (by annotation count):")
        for cid, count in cat_counter.most_common(10):
            cname = cat_by_id[cid]["name"]
            print(f"  {cname} (id={cid}): {count}")

    if missing_files:
        print("Missing image files (first 10):")
        for p in missing_files[:10]:
            print(f"  {p}")

    if bad_anns:
        print(f"Invalid annotations referencing missing image/category: {len(bad_anns)} (showing first 10)")
        print(bad_anns[:10])

    ok = (len(missing_files) == 0 and len(bad_anns) == 0)

    # Optional: visualize random samples with bbox overlays
    if sample_vis > 0 and anns:
        # Build per-image annotations for quick sampling
        anns_by_img = defaultdict(list)
        for a in anns:
            anns_by_img[a["image_id"]].append(a)

        sample_images = random.sample(images, k=min(sample_vis, len(images)))
        out_dir = Path("outputs/vis")
        out_dir.mkdir(parents=True, exist_ok=True)
        for im in sample_images:
            img_path = images_dir / im["file_name"]
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            for a in anns_by_img.get(im["id"], []):
                # Draw bbox if present
                if "bbox" in a and isinstance(a["bbox"], (list, tuple)) and len(a["bbox"]) == 4:
                    x, y, w, h = a["bbox"]
                    pt1 = (int(x), int(y))
                    pt2 = (int(x + w), int(y + h))
                    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
                    cid = a.get("category_id")
                    cname = cat_by_id.get(cid, {}).get("name", str(cid))
                    cv2.putText(img, cname, (pt1[0], max(0, pt1[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            out_fp = out_dir / f"vis_{Path(im['file_name']).stem}.jpg"
            cv2.imwrite(str(out_fp), img)
        print(f"Saved {len(sample_images)} visualization(s) to {out_dir}")

    return ok
