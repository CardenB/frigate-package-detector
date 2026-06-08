#!/usr/bin/env python3
"""Download a COCO-2017 subset via FiftyOne into data/raw/coco/.

This is the anti-forgetting set: it carries the COCO classes Frigate relies on
(person, car, dog, ...) through fine-tuning so they aren't lost. Skipped in
single-class (package-only) builds.
"""
from __future__ import annotations

import shutil

from common import RAW, build_mode, datasets_cfg


def main() -> None:
    if build_mode() == "single":
        print("[coco] BUILD_MODE=single — package-only build, skipping COCO")
        return

    cfg = datasets_cfg().get("coco_subset", {})
    if not cfg.get("enabled", True):
        print("[coco] disabled in datasets.yaml — skipping")
        return

    import fiftyone as fo
    import fiftyone.zoo as foz

    # Retain all 80 COCO classes (drop-in for default model) unless a subset is
    # explicitly requested.
    all_classes = cfg.get("all_coco_classes", True)
    classes = None if all_classes else cfg.get("classes")
    max_samples = cfg.get("max_samples", {})

    out = RAW / "coco"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[coco] classes={'ALL 80' if classes is None else classes}")
    for split in ("train", "validation"):
        load_kwargs = dict(
            split=split,
            label_types=["detections"],
            max_samples=max_samples.get(split),
            dataset_name=f"coco-{split}-subset",
            drop_existing_dataset=True,
        )
        if classes is not None:
            load_kwargs["classes"] = classes
        ds = foz.load_zoo_dataset("coco-2017", **load_kwargs)

        view = ds
        export_classes = classes
        if classes is not None:
            view = ds.filter_labels(
                "ground_truth", fo.ViewField("label").is_in(classes),
                only_matches=True)
        else:
            export_classes = ds.default_classes  # full COCO-80 ordering

        split_out = out / split
        view.export(
            export_dir=str(split_out),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=export_classes,
            split="val" if split == "validation" else "train",
        )
        print(f"[coco] exported {split} -> {split_out}")

    print(f"[coco] ready -> {out}")


if __name__ == "__main__":
    main()
