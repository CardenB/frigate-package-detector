#!/usr/bin/env python3
"""Pull recent Frigate detections into a review round.

Fetches events from the Frigate HTTP API, downloads each full-frame snapshot,
and writes Frigate's own detected box as a CANDIDATE YOLO label — so review is
verification, not from-scratch labeling. Output:

    data/review/incoming/<round>/
        images/<event_id>.jpg
        labels/<event_id>.txt        # candidate boxes (unified class ids)
        manifest.jsonl               # event metadata (camera, score, ...)

Config defaults come from datasets.yaml: frigate.pull; override on the CLI.
Needs FRIGATE_URL in .env (deployment-private, e.g. http://10.0.0.5:5000).
Optional FRIGATE_API_KEY for a bearer token.

Usage:
    python scripts/frigate_pull.py --check          # test connectivity only
    python scripts/frigate_pull.py --limit 50       # pull 50 package events
    python scripts/frigate_pull.py --labels package,person --after 2026-06-01
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import requests

from common import DATA, class_index, datasets_cfg

REVIEW_INCOMING = DATA / "review" / "incoming"


def api_base() -> str:
    url = os.environ.get("FRIGATE_URL", "").rstrip("/")
    if not url:
        sys.exit("FRIGATE_URL not set. Put it in .env (e.g. http://10.0.0.5:5000).")
    return url


def session() -> requests.Session:
    s = requests.Session()
    key = os.environ.get("FRIGATE_API_KEY")
    if key:
        s.headers["Authorization"] = f"Bearer {key}"
    return s


def to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(s)  # already a unix ts
    except ValueError:
        return dt.datetime.fromisoformat(s).timestamp()


def candidate_box(ev: dict) -> list[float] | None:
    """Frigate box -> YOLO (cx,cy,w,h) normalized. Frigate gives [x,y,w,h] as
    top-left + size ratios of the full frame (under event['data']['box'])."""
    box = (ev.get("data") or {}).get("box") or ev.get("box")
    if not box or len(box) != 4:
        return None
    x, y, w, h = (float(v) for v in box)
    cx, cy = x + w / 2, y + h / 2
    out = [cx, cy, w, h]
    if any(v < 0 or v > 1 for v in out) or w <= 0 or h <= 0:
        return None
    return out


def main() -> None:
    cfg = (datasets_cfg().get("frigate") or {}).get("pull", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="test connectivity and exit")
    ap.add_argument("--labels", default=",".join(cfg.get("labels", ["package"])))
    ap.add_argument("--cameras", default="", help="comma-separated; default all")
    ap.add_argument("--limit", type=int, default=cfg.get("limit", 200))
    ap.add_argument("--min-score", type=float, default=cfg.get("min_score", 0.0))
    ap.add_argument("--after", default=None, help="ISO date or unix ts")
    ap.add_argument("--before", default=None, help="ISO date or unix ts")
    ap.add_argument("--round", default=None, help="round id (default: timestamp)")
    args = ap.parse_args()

    base, s = api_base(), session()

    if args.check:
        r = s.get(f"{base}/api/version", timeout=10)
        r.raise_for_status()
        print(f"[pull] connected to Frigate {r.text.strip()} at {base}")
        return

    round_id = args.round or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = REVIEW_INCOMING / round_id
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    params = {
        "labels": args.labels,
        "limit": args.limit,
        "has_snapshot": 1,
        "include_thumbnails": 0,
    }
    if args.cameras:
        params["cameras"] = args.cameras
    if (a := to_epoch(args.after)):
        params["after"] = a
    if (b := to_epoch(args.before)):
        params["before"] = b

    print(f"[pull] GET {base}/api/events  {params}")
    r = s.get(f"{base}/api/events", params=params, timeout=30)
    r.raise_for_status()
    events = r.json()
    print(f"[pull] {len(events)} events returned")

    cidx = class_index()
    kept = no_box = low = 0
    with open(out / "manifest.jsonl", "w") as mf:
        for ev in events:
            score = ev.get("data", {}).get("score") or ev.get("top_score") or 0
            if score < args.min_score:
                low += 1
                continue
            eid = ev["id"]
            # full frame, no drawn box, so candidate ratios map cleanly
            snap = s.get(f"{base}/api/events/{eid}/snapshot.jpg",
                         params={"bbox": 0, "crop": 0}, timeout=30)
            if snap.status_code != 200:
                continue
            (out / "images" / f"{eid}.jpg").write_bytes(snap.content)

            lines = []
            box = candidate_box(ev)
            cid = cidx.get(ev.get("label", ""))
            if box is None or cid is None:
                no_box += 1  # snapshot saved, but no usable candidate -> label by hand
            else:
                lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in box))
            (out / "labels" / f"{eid}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "")

            mf.write(json.dumps({
                "id": eid, "camera": ev.get("camera"), "label": ev.get("label"),
                "score": round(float(score), 4), "start_time": ev.get("start_time"),
            }) + "\n")
            kept += 1

    print(f"[pull] saved {kept} frames -> {out}")
    print(f"[pull]   {no_box} had no usable candidate box (label by hand in review)")
    if low:
        print(f"[pull]   {low} skipped below --min-score {args.min_score}")
    print(f"[pull] next: python scripts/frigate_review.py --round {round_id}")


if __name__ == "__main__":
    main()
