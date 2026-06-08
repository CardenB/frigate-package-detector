#!/usr/bin/env python3
"""Download an Open Images V7 subset via FiftyOne into data/raw/open_images/.

Pulls only the classes we care about, capped per-class (see datasets.yaml), and
exports to YOLOv5 format (images/ + labels/ + dataset.yaml). "Box" is included
as a NOISY package proxy unless open_images.map_box is false.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from common import RAW, datasets_cfg


def main() -> None:
    cfg = datasets_cfg().get("open_images", {})
    if not cfg.get("enabled", True):
        print("[open_images] disabled in datasets.yaml — skipping")
        return

    import fiftyone as fo
    import fiftyone.zoo as foz

    caps: dict[str, int] = dict(cfg.get("max_samples_per_class", {}))
    if not cfg.get("map_box", True):
        caps.pop("Box", None)
    splits = cfg.get("splits", ["train", "validation"])

    out = RAW / "open_images"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    all_classes = sorted(caps)
    print(f"[open_images] classes={all_classes} splits={splits}")

    for split in splits:
        # FiftyOne caps samples globally, not per-class, so to honor per-class
        # caps we load each class separately then merge.
        merged = None
        for cls, cap in caps.items():
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=[cls],
                max_samples=cap,
                dataset_name=f"oiv7-{split}-{cls}".replace(" ", "_"),
                drop_existing_dataset=True,
            )
            merged = ds.clone() if merged is None else merged.merge_samples(ds) or merged

        # Restrict detections to our target classes and export YOLOv5.
        view = merged.filter_labels(
            "ground_truth", fo.ViewField("label").is_in(all_classes), only_matches=True
        )
        split_out = out / split
        view.export(
            export_dir=str(split_out),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=all_classes,
            split="val" if split == "validation" else "train",
        )
        print(f"[open_images] exported {split} -> {split_out}")

    print(f"[open_images] ready -> {out}")


if __name__ == "__main__":
    main()
