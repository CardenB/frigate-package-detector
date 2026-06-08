#!/usr/bin/env python3
"""Normalize every raw source into data/interim/<source>/{images,labels}.

- Remaps each source's native class ids -> unified ids (configs/classes.yaml).
- Drops any class not in the active build's class set.
- Converts segmentation polygons -> axis-aligned bboxes.
- Flattens splits; build_dataset.py re-splits later.

Re-run this any time you edit classes.yaml — it's fast and needs no network.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml

from common import INTERIM, RAW, Remapper, curation, datasets_cfg, unified_classes

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_image(label_path: Path) -> Path | None:
    """Image that pairs with a YOLO label .txt (labels/ <-> images/).

    Strip the literal '.txt' rather than using Path.with_suffix: many sources use
    filenames with extra dots (e.g. `name_jpg.rf.<hash>.txt`), and with_suffix
    would mangle everything after the last dot (-> `name_jpg.rf.jpg`).
    """
    parts = list(label_path.parts)
    if "labels" in parts:
        parts[len(parts) - 1 - parts[::-1].index("labels")] = "images"
    base = str(Path(*parts))
    if base.endswith(".txt"):
        base = base[:-4]
    for ext in IMG_EXTS:
        cand = Path(base + ext)
        if cand.exists():
            return cand
    return None


def poly_or_box_to_xywh(coords: list[float]) -> tuple[float, float, float, float] | None:
    """Return normalized (cx,cy,w,h). Accepts a 4-tuple bbox or a polygon."""
    if len(coords) == 4:
        return tuple(coords)  # already cx,cy,w,h
    if len(coords) >= 6 and len(coords) % 2 == 0:
        xs = coords[0::2]
        ys = coords[1::2]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
    return None


def ingest_unit(root: Path, native_names: list[str], remap: Remapper,
                source: str, out_img: Path, out_lbl: Path,
                keep_negatives: bool = False) -> tuple[int, int, int]:
    """Process one (images,labels) tree with a known native class list.

    Returns (images, boxes, negatives). When keep_negatives is set, an image
    whose label file exists but yields no boxes is kept as a hard negative
    (empty label) instead of being skipped — used for reviewed Frigate
    false-positive frames.
    """
    kept_imgs = kept_boxes = kept_neg = 0
    # os.walk with followlinks=True: some sources (e.g. package_seg) symlink
    # their images/labels dirs, and Path.rglob does NOT descend into symlinked
    # directories — using rglob here would silently drop those datasets.
    label_files = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for fn in filenames:
            if fn.endswith(".txt"):
                label_files.append(Path(dirpath) / fn)
    label_files.sort()
    for lf in label_files:
        if "labels" not in lf.parts or lf.name == "_names.txt":
            continue
        img = find_image(lf)
        if img is None:
            continue

        out_lines: list[str] = []
        for line in lf.read_text().splitlines():
            tok = line.split()
            if len(tok) < 5:
                continue
            try:
                native_id = int(float(tok[0]))
                coords = [float(t) for t in tok[1:]]
            except ValueError:
                continue
            if native_id < 0 or native_id >= len(native_names):
                continue
            new_id = remap.to_id(native_names[native_id])
            if new_id is None:
                continue  # dropped class
            box = poly_or_box_to_xywh(coords)
            if box is None:
                continue
            cx, cy, w, h = box
            if w <= 0 or h <= 0:
                continue
            out_lines.append(f"{new_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not out_lines and not keep_negatives:
            continue  # skip images with no surviving labels (unless keeping negs)

        # Unique, readable stem from the path under data/raw/ so files from
        # different projects/splits (e.g. two roboflow projects each with
        # train/labels/img1.txt) never collide.
        rel = lf.relative_to(RAW).with_suffix("")
        stem = "__".join(rel.parts)
        # Empty file (no trailing newline) is a valid YOLO hard negative.
        (out_lbl / f"{stem}.txt").write_text(
            ("\n".join(out_lines) + "\n") if out_lines else "")
        dst_img = out_img / f"{stem}{img.suffix.lower()}"
        if not dst_img.exists():
            try:
                dst_img.symlink_to(img.resolve())
            except OSError:
                shutil.copy2(img, dst_img)
        kept_imgs += 1
        kept_boxes += len(out_lines)
        if not out_lines:
            kept_neg += 1
    return kept_imgs, kept_boxes, kept_neg


def names_from_yaml(p: Path) -> list[str]:
    data = yaml.safe_load(p.read_text())
    names = data["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def units_for_source(source: str) -> list[tuple[Path, list[str]]]:
    """Discover (root, native_names) pairs for a raw source."""
    root = RAW / source
    if not root.exists():
        return []

    if source == "package_seg":
        names = (root / "_names.txt").read_text().split()
        return [(root, names)]

    if source in ("open_images", "coco", "frigate"):
        # frigate rounds live in round-*/ each with a dataset.yaml (same layout).
        units = []
        for yml in root.rglob("dataset.yaml"):
            units.append((yml.parent, names_from_yaml(yml)))
        return units

    if source == "roboflow":
        units = []
        for yml in root.rglob("data.yaml"):
            units.append((yml.parent, names_from_yaml(yml)))
        return units

    return []


def main() -> None:
    only = sys.argv[1:]  # optional: restrict to named sources
    sources = ["package_seg", "open_images", "coco", "roboflow", "frigate"]
    if only:
        sources = [s for s in sources if s in only]

    print(f"[convert] build classes = {unified_classes()}")
    grand_i = grand_b = grand_n = 0
    for source in sources:
        # Source-level curation (roboflow is curated per-project below).
        if source != "roboflow":
            status, reason = curation(source)
            if status in ("deny", "defer"):
                # Remove any stale interim from before it was excluded, so
                # build_dataset.py doesn't keep pooling it.
                if (INTERIM / source).exists():
                    shutil.rmtree(INTERIM / source)
                    print(f"[convert] {source}: removed stale interim")
                label = "BLACKLISTED" if status == "deny" else "DEFERRED (left out for now)"
                print(f"[convert] {source}: {label} — skipping ({reason[:60]})")
                continue
            if status == "pending":
                print(f"[convert] {source}: ⚠ PENDING review (not whitelisted)")

        units = units_for_source(source)
        if not units:
            print(f"[convert] {source}: no raw data found — skipping")
            continue

        if source == "roboflow":
            kept = []
            for root, names in units:
                status, reason = curation(root.name)
                if status in ("deny", "defer"):
                    label = "BLACKLISTED" if status == "deny" else "DEFERRED"
                    print(f"[convert] roboflow/{root.name}: {label} — skipping ({reason[:55]})")
                    continue
                if status == "pending":
                    print(f"[convert] roboflow/{root.name}: ⚠ PENDING review")
                kept.append((root, names))
            units = kept
            if not units:
                print("[convert] roboflow: all projects skipped")
                continue

        try:
            remap = Remapper(source)
        except KeyError as e:
            print(f"[convert] {source}: {e} — skipping")
            continue

        out_img = INTERIM / source / "images"
        out_lbl = INTERIM / source / "labels"
        if (INTERIM / source).exists():
            shutil.rmtree(INTERIM / source)
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        keep_neg = bool(datasets_cfg().get(source, {}).get("keep_negatives", False))
        si = sb = sn = 0
        for root, native_names in units:
            i, b, n = ingest_unit(root, native_names, remap, source,
                                  out_img, out_lbl, keep_negatives=keep_neg)
            si += i
            sb += b
            sn += n
        neg_note = f", {sn} negatives" if keep_neg else ""
        print(f"[convert] {source}: {si} images, {sb} boxes{neg_note} -> {INTERIM/source}")
        grand_i += si
        grand_b += sb
        grand_n += sn

    print(f"[convert] TOTAL: {grand_i} images, {grand_b} boxes, {grand_n} negatives")


if __name__ == "__main__":
    main()
