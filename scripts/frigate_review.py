#!/usr/bin/env python3
"""Review a pulled Frigate round in FiftyOne (verify-first).

Loads data/review/incoming/<round>/ into a persistent FiftyOne dataset with
Frigate's candidate boxes pre-drawn, and opens the App. You then TAG each sample:

    good      -> keep Frigate's box(es) as the label
    negative  -> false positive / nothing there -> becomes a HARD NEGATIVE
    holdout   -> verified-correct, but RESERVE for evaluation (never trained on)
    fix       -> box is wrong/missing -> deferred (edit later via CVAT)

(untagged samples are treated as not-yet-reviewed and skipped on export.)

Reserve ~10-20% of clean frames as `holdout` so you build a stable gold eval
set (data/gold_eval/) and can measure real porch performance each round.

Tagging in the App: select samples (click, or shift-click ranges) then press
the tag button / use the tag sidebar to add one of the tags above.

Usage:
    python scripts/frigate_review.py --round <id>     # specific round
    python scripts/frigate_review.py                  # most recent round
"""
from __future__ import annotations

import argparse
import json
import sys

from common import DATA, unified_classes

REVIEW_INCOMING = DATA / "review" / "incoming"
TAGS = ("good", "negative", "holdout", "fix")


def latest_round() -> str | None:
    rounds = sorted(p.name for p in REVIEW_INCOMING.glob("*") if p.is_dir())
    return rounds[-1] if rounds else None


def load_round(round_id: str):
    import fiftyone as fo

    root = REVIEW_INCOMING / round_id
    if not root.exists():
        sys.exit(f"No such round: {root}")
    names = unified_classes()

    scores = {}
    mf = root / "manifest.jsonl"
    if mf.exists():
        for line in mf.read_text().splitlines():
            d = json.loads(line)
            scores[d["id"]] = d

    name = f"frigate-{round_id}"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    ds = fo.Dataset(name=name, persistent=True)
    ds.info["round_id"] = round_id

    samples = []
    for img in sorted((root / "images").glob("*.jpg")):
        eid = img.stem
        s = fo.Sample(filepath=str(img.resolve()))
        meta = scores.get(eid, {})
        s["camera"] = meta.get("camera")
        s["frigate_score"] = meta.get("score")
        dets = []
        lbl = root / "labels" / f"{eid}.txt"
        for line in (lbl.read_text().splitlines() if lbl.exists() else []):
            t = line.split()
            if len(t) != 5:
                continue
            cid, cx, cy, w, h = int(t[0]), *map(float, t[1:])
            dets.append(fo.Detection(
                label=names[cid] if cid < len(names) else str(cid),
                bounding_box=[cx - w / 2, cy - h / 2, w, h]))
        s["frigate"] = fo.Detections(detections=dets)
        samples.append(s)
    ds.add_samples(samples)
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default=None)
    ap.add_argument("--port", type=int, default=5151)
    args = ap.parse_args()

    import fiftyone as fo

    round_id = args.round or latest_round()
    if not round_id:
        sys.exit(f"No rounds found in {REVIEW_INCOMING}. Run frigate_pull.py first.")

    ds = load_round(round_id)
    n_box = sum(1 for s in ds if s["frigate"].detections)
    print(f"[review] round {round_id}: {len(ds)} frames "
          f"({n_box} with a candidate box, {len(ds) - n_box} need manual box)")
    print(f"[review] TAG each frame one of: {', '.join(TAGS)}")
    print(f"[review] when done, close the App, then:")
    print(f"[review]   python scripts/frigate_export.py --round {round_id}")

    session = fo.launch_app(ds, port=args.port)
    session.wait()  # blocks until you close the App


if __name__ == "__main__":
    main()
