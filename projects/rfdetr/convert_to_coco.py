#!/usr/bin/env python3
"""Convert the repo's YOLO dataset (data/final) -> a LEAN COCO-JSON set for RF-DETR.

RF-DETR/DETR is slow to train and shines on small focused datasets, so we DON'T
mirror yolov9's full 81-class / 133k-image set. Instead:
  - keep only the classes we actually track (`security_classes` in classes.yaml),
    re-indexed to a CONTIGUOUS 1..K category space (RF-DETR wants no gaps);
  - keep ALL `package` images (the signal), and cap the COCO-replay images per
    class (--cap-per-class) so the set stays ~20-30k, not 133k.

Writes projects/rfdetr/data/{train,valid,test}/ (images symlinked + COCO json)
and data/classes.txt (the K-class order = the deploy labelmap order).

    python convert_to_coco.py                       # all splits, lean+capped
    python convert_to_coco.py --split test --limit 50   # quick smoke
    python convert_to_coco.py --keep-classes person,package --cap-per-class 1000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FINAL = REPO / "data" / "final"
OUT = HERE / "data"
SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PACKAGE = "package"


def classes_cfg() -> dict:
    return yaml.safe_load((REPO / "configs" / "classes.yaml").read_text())


def find_image(label_path: Path) -> Path | None:
    base = Path(str(label_path).replace("/labels/", "/images/")).with_suffix("")
    for ext in IMG_EXTS:
        if (c := base.with_suffix(ext)).exists():
            return c
    return None


def read_kept_boxes(lf: Path, uid_to_cat: dict[int, int]):
    """Return [(cat_id, cx,cy,w,h)] for kept classes only; [] if none."""
    out = []
    for line in lf.read_text().splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        uid = int(float(t[0]))
        if uid in uid_to_cat:
            out.append((uid_to_cat[uid], *(float(x) for x in t[1:5])))
    return out


def convert_split(yolo_split, keep_names, uid_to_cat, pkg_cat, cap, limit, seed):
    rf_split = SPLIT_MAP[yolo_split]
    lbl_dir = FINAL / yolo_split / "labels"
    out_dir = OUT / rf_split
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = [{"id": i + 1, "name": n, "supercategory": "none"}
                  for i, n in enumerate(keep_names)]

    # Pass 1: gather candidates (image has >=1 kept box).
    cands = []
    for lf in sorted(lbl_dir.glob("*.txt")):
        boxes = read_kept_boxes(lf, uid_to_cat)
        if boxes:
            cands.append((lf, boxes))
    random.Random(seed).shuffle(cands)  # deterministic, source-mixing order
    if limit:
        cands = cands[:limit]

    # Pass 2: greedy per-class cap; package images always kept.
    per_cat = {c["id"]: 0 for c in categories}
    images, annotations = [], []
    img_id = ann_id = 0
    for lf, boxes in cands:
        cats = {b[0] for b in boxes}
        is_pkg = pkg_cat in cats
        if cap and not is_pkg and all(per_cat[c] >= cap for c in cats):
            continue  # every class in this image already capped
        img = find_image(lf)
        if img is None:
            continue
        try:
            with Image.open(img) as im:
                w, h = im.size
        except Exception:
            continue
        img_id += 1
        for c in cats:
            per_cat[c] += 1
        fname = f"{lf.stem}{img.suffix.lower()}"
        dst = out_dir / fname
        if not dst.exists():
            try:
                dst.symlink_to(img.resolve())
            except OSError:
                import shutil
                shutil.copy2(img, dst)
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})
        for cat_id, cx, cy, bw, bh in boxes:
            x, y, pw, ph = (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h
            if pw <= 0 or ph <= 0:
                continue
            ann_id += 1
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cat_id,
                "bbox": [round(x, 2), round(y, 2), round(pw, 2), round(ph, 2)],
                "area": round(pw * ph, 2), "iscrowd": 0})

    (out_dir / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations,
                    "categories": categories}))
    print(f"[coco] {yolo_split}->{rf_split}: {len(images)} imgs, "
          f"{len(annotations)} anns  per-class={per_cat}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLIT_MAP), help="one split (default: all)")
    ap.add_argument("--keep-classes", default=None,
                    help="comma list (default: security_classes from classes.yaml)")
    ap.add_argument("--cap-per-class", type=int, default=1500,
                    help="max COCO-replay images per class (package always kept); 0=no cap")
    ap.add_argument("--limit", type=int, default=None, help="cap candidates (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not FINAL.exists():
        raise SystemExit(f"{FINAL} not found — run `make data` in the repo first.")
    cfg = classes_cfg()
    uid_of = {n: i for i, n in enumerate(cfg["classes"])}     # name -> unified id
    keep_names = ([c.strip() for c in args.keep_classes.split(",")]
                  if args.keep_classes else list(cfg["security_classes"]))
    keep_names = [n for n in keep_names if n in uid_of]       # validate
    uid_to_cat = {uid_of[n]: i + 1 for i, n in enumerate(keep_names)}  # -> 1..K
    pkg_cat = uid_to_cat.get(uid_of.get(PACKAGE, -1))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "classes.txt").write_text("\n".join(keep_names) + "\n")
    print(f"[coco] {len(keep_names)} classes (lean): {keep_names}")
    print(f"[coco] cap-per-class={args.cap_per_class}  package always kept")
    for s in ([args.split] if args.split else list(SPLIT_MAP)):
        convert_split(s, keep_names, uid_to_cat, pkg_cat, args.cap_per_class,
                      args.limit, args.seed)


if __name__ == "__main__":
    main()
