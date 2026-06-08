#!/usr/bin/env python3
"""Export a reviewed Frigate round to a gold YOLO dataset.

Reads the persistent FiftyOne dataset from frigate_review.py, applies your tags,
and writes:

    data/raw/frigate/round-<id>/      (TRAINING data — folded in by convert/build)
        images/...            (copied, self-contained gold data)
        labels/...            (YOLO; empty file == hard negative)
        dataset.yaml          (unified class names, so convert_to_yolo picks it up)

    data/gold_eval/                   (EVAL ONLY — never trained on; accumulates)
        images/ labels/ dataset.yaml  (frames you tagged `holdout`)

Tag handling:
    good      -> export box(es) as labels         (-> round, training)
    negative  -> empty label (hard negative)       (-> round, training)
    holdout   -> verified frame RESERVED for eval   (-> data/gold_eval, NOT trained)
    fix       -> skipped (needs box editing first)
    untagged  -> skipped (not reviewed)

If a sample has a non-empty `ground_truth` field (e.g. corrected via CVAT), that
is used instead of Frigate's original boxes.

Usage:
    python scripts/frigate_export.py --round <id>
    python scripts/frigate_export.py                 # most recent round
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

from common import GOLD_EVAL, RAW, unified_classes
from frigate_review import latest_round


def write_dataset_yaml(path: Path, names: list[str], extra: dict | None = None) -> None:
    doc = {"names": {i: n for i, n in enumerate(names)}}
    if extra:
        doc.update(extra)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def dets_of(sample):
    """Prefer corrected ground_truth boxes if present, else Frigate's."""
    for field in ("ground_truth", "frigate"):
        d = sample.get_field(field) if sample.has_field(field) else None
        if d and d.detections:
            return d.detections
    return []


def boxes_to_lines(sample, cidx) -> list[str]:
    """Sample detections -> YOLO label lines (unified class ids, cx cy w h)."""
    lines = []
    for d in dets_of(sample):
        cid = cidx.get(d.label)
        if cid is None:
            continue
        tlx, tly, w, h = d.bounding_box
        lines.append(f"{cid} {tlx + w/2:.6f} {tly + h/2:.6f} {w:.6f} {h:.6f}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default=None)
    args = ap.parse_args()

    import fiftyone as fo

    round_id = args.round or latest_round()
    if not round_id:
        sys.exit("No round specified and none found.")
    name = f"frigate-{round_id}"
    if not fo.dataset_exists(name):
        sys.exit(f"FiftyOne dataset '{name}' not found. Run frigate_review.py first.")
    ds = fo.load_dataset(name)

    names = unified_classes()
    cidx = {n: i for i, n in enumerate(names)}

    # Training round (good/negative) — rebuilt fresh each export.
    out = RAW / "frigate" / f"round-{round_id}"
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)

    # Gold eval set (holdout) — ACCUMULATES across rounds. Clear only this
    # round's prior contribution so re-exporting a round is idempotent.
    (GOLD_EVAL / "images").mkdir(parents=True, exist_ok=True)
    (GOLD_EVAL / "labels").mkdir(parents=True, exist_ok=True)
    for d in ("images", "labels"):
        for f in (GOLD_EVAL / d).glob(f"{round_id}__*"):
            f.unlink()

    def emit(dest, stem, src, lines):
        shutil.copy2(src, dest / "images" / f"{stem}.jpg")
        (dest / "labels" / f"{stem}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "")

    n_pos = n_neg = n_fix = n_skip = n_box = n_gold = n_gold_box = 0
    for s in ds:
        tags = set(s.tags)
        if "fix" in tags:
            n_fix += 1
            continue
        is_holdout = "holdout" in tags
        if not is_holdout and "good" not in tags and "negative" not in tags:
            n_skip += 1
            continue

        stem = f"{round_id}__{Path(s.filepath).stem}"
        # holdout & good carry boxes; negative is an empty (hard-negative) label.
        lines = [] if (not is_holdout and "good" not in tags) else boxes_to_lines(s, cidx)

        if is_holdout:
            emit(GOLD_EVAL, stem, s.filepath, lines)
            n_gold += 1
            n_gold_box += len(lines)
        else:
            emit(out, stem, s.filepath, lines)
            if lines:
                n_pos += 1
                n_box += len(lines)
            else:
                n_neg += 1  # negative, or 'good' with no usable box

    write_dataset_yaml(out / "dataset.yaml", names)

    # Refresh the gold dataset.yaml (val/test point at the flat images dir).
    gold_imgs = list((GOLD_EVAL / "images").glob("*.jpg"))
    if gold_imgs:
        write_dataset_yaml(GOLD_EVAL / "dataset.yaml", names,
                           {"path": str(GOLD_EVAL), "val": "images", "test": "images"})

    print(f"[export] round {round_id}")
    print(f"[export]   TRAIN -> {out}")
    print(f"[export]     {n_pos} positives ({n_box} boxes), {n_neg} hard negatives")
    print(f"[export]   GOLD  -> {GOLD_EVAL}  (+{n_gold} this round, {len(gold_imgs)} total)")
    print(f"[export]   skipped: {n_fix} need-fix, {n_skip} unreviewed")
    print(f"[export] next: make convert build   (then make benchmark for gold P/R)")


if __name__ == "__main__":
    main()
