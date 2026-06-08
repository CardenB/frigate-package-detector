#!/usr/bin/env python3
"""Download the Ultralytics package-seg dataset into data/raw/package_seg/.

It's a segmentation set; labels are normalized polygons. We keep them as-is here
and convert polygons -> bboxes in convert_to_yolo.py.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from common import RAW, datasets_cfg
from ultralytics.data.utils import check_det_dataset


def main() -> None:
    cfg = datasets_cfg().get("package_seg", {})
    if not cfg.get("enabled", True):
        print("[package_seg] disabled in datasets.yaml — skipping")
        return

    out = RAW / "package_seg"
    out.mkdir(parents=True, exist_ok=True)

    # Triggers download into the ultralytics datasets dir and returns paths.
    info = check_det_dataset("package-seg.yaml", autodownload=True)
    root = Path(info["path"])
    names = info["names"]  # {0: 'package'}
    print(f"[package_seg] source at {root}, names={names}")

    # Mirror the train/val image+label dirs into our raw tree (symlink to save
    # disk; copytree fallback if symlinks are unavailable on this FS).
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            src = root / kind / split
            if not src.exists():
                continue
            dst = out / kind / split
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                continue
            try:
                dst.symlink_to(src, target_is_directory=True)
            except OSError:
                shutil.copytree(src, dst)

    # Record native class names (index-ordered) for the converter.
    names_list = [names[i] for i in sorted(names)]
    (out / "_names.txt").write_text("\n".join(names_list) + "\n")
    print(f"[package_seg] ready -> {out}  (labels are POLYGON/segmentation)")


if __name__ == "__main__":
    main()
