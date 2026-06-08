#!/usr/bin/env python3
"""Report per-class box counts and image counts across data/final splits.

Use this before training: if `package` has far fewer boxes than `person`, the
model will under-detect it. Remedies: pull more Roboflow package projects, raise
Open Images "Box" cap, or lower the COCO caps in datasets.yaml.
"""
from __future__ import annotations

from collections import Counter

from common import FINAL, unified_classes


def main() -> None:
    names = unified_classes()
    print(f"classes: {names}\n")
    grand = Counter()
    for split in ("train", "val", "test"):
        lbl_dir = FINAL / split / "labels"
        if not lbl_dir.exists():
            continue
        boxes = Counter()
        n_imgs = 0
        for txt in lbl_dir.glob("*.txt"):
            n_imgs += 1
            for line in txt.read_text().splitlines():
                if line.strip():
                    boxes[int(line.split()[0])] += 1
        grand.update(boxes)
        print(f"== {split} ==  ({n_imgs} images)")
        for i, name in enumerate(names):
            print(f"  {name:<12} {boxes.get(i, 0):>8}")
        print()

    total = sum(grand.values()) or 1
    print("== ALL SPLITS — share of boxes ==")
    for i, name in enumerate(names):
        c = grand.get(i, 0)
        bar = "#" * int(40 * c / total)
        print(f"  {name:<12} {c:>8}  {100*c/total:5.1f}%  {bar}")

    pkg = grand.get(names.index("package"), 0) if "package" in names else 0
    if "package" in names and pkg < total * 0.05:
        print("\n[warn] package is <5% of all boxes — consider adding more "
              "package data or trimming COCO/OpenImages caps in datasets.yaml.")


if __name__ == "__main__":
    main()
