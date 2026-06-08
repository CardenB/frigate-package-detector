#!/usr/bin/env python3
"""Build an HTML review report: up to N sample images per unified class, per
dataset, with that class's boxes drawn (under our remapping). Lets you eyeball
label quality and verify the class mapping (e.g. Open Images "Box" -> package)
before training.

Usage:
    python scripts/make_review_report.py [N]      # default N=5
Open report/index.html in a browser afterwards.
"""
from __future__ import annotations

import html
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import convert_to_yolo as C
from common import RAW, REPO, Remapper, curation, unified_classes

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
THUMB = 480
SEED = 0
OUT = REPO / "report"
IMGS = OUT / "imgs"
NAMES = unified_classes()

try:
    FONT = ImageFont.load_default(16)
except TypeError:
    FONT = ImageFont.load_default()


def _tag(name: str) -> str | None:
    """Display suffix for curation status; None means 'drop it (blacklisted)'."""
    status, _ = curation(name)
    return {
        "deny": None,                  # blacklisted — drop from report
        "defer": " ⏸ deferred",        # left out of builds, kept visible to revisit
        "pending": " ⚠ PENDING",
        "allow": " ✓ whitelisted",
    }[status]


def dataset_entries() -> list[tuple[str, str, list]]:
    """(display_label, remap_source, [(root, native_names), ...]) per dataset.

    Blacklisted datasets are dropped; others get a status tag in the heading.
    """
    entries = []
    for src in ("package_seg", "open_images", "coco"):
        if not (RAW / src).exists():
            continue
        tag = _tag(src)
        if tag is None:
            continue
        u = C.units_for_source(src)
        if u:
            entries.append((f"{src}{tag}", src, u))
    for root, names in C.units_for_source("roboflow"):
        tag = _tag(root.name)
        if tag is None:
            continue
        entries.append((f"roboflow: {root.name}{tag}", "roboflow", [(root, names)]))
    return entries


def label_files(units) -> list[tuple[Path, list]]:
    out = []
    for root, names in units:
        for dp, _, fns in os.walk(root, followlinks=True):  # follow symlinked dirs
            for fn in fns:
                if fn.endswith(".txt") and fn != "_names.txt":
                    lf = Path(dp) / fn
                    if "labels" in lf.parts:
                        out.append((lf, names))
    return out


def sample(units, source) -> dict[int, list]:
    """unified_class_id -> [(image_path, [boxes]), ...] (<= N each)."""
    remap = Remapper(source)
    items = label_files(units)
    random.Random(SEED).shuffle(items)
    buckets: dict[int, list] = defaultdict(list)
    for lf, names in items:
        img = C.find_image(lf)
        if img is None:
            continue
        per_class = defaultdict(list)
        for line in lf.read_text().splitlines():
            tok = line.split()
            if len(tok) < 5:
                continue
            try:
                nid = int(float(tok[0]))
                coords = [float(t) for t in tok[1:]]
            except ValueError:
                continue
            if nid < 0 or nid >= len(names):
                continue
            uid = remap.to_id(names[nid])
            if uid is None:
                continue
            box = C.poly_or_box_to_xywh(coords)
            if box and box[2] > 0 and box[3] > 0:
                per_class[uid].append(box)
        for uid, boxes in per_class.items():
            if len(buckets[uid]) < N:
                buckets[uid].append((img, boxes))
    return buckets


def render(img_path: Path, boxes, out_path: Path) -> bool:
    try:
        im = Image.open(img_path).convert("RGB")
    except Exception:
        return False
    W, H = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, int(min(W, H) * 0.006))
    for cx, cy, w, h in boxes:
        x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
        x1, y1 = (cx + w / 2) * W, (cy + h / 2) * H
        d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=lw)
    scale = THUMB / max(W, H)
    if scale < 1:
        im = im.resize((max(1, int(W * scale)), max(1, int(H * scale))))
    im.save(out_path, quality=85)
    return True


def slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


def main() -> None:
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    IMGS.mkdir(parents=True, exist_ok=True)

    entries = dataset_entries()
    print(f"[report] datasets: {[e[0] for e in entries]}  (N={N} per class)")

    parts = [
        "<!doctype html><meta charset=utf-8><title>Dataset review</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}"
        "h2{border-bottom:2px solid #444;padding-bottom:4px;margin-top:40px}"
        "h3{color:#9cf;margin:18px 0 6px}.row{display:flex;flex-wrap:wrap;gap:8px}"
        "img{border:1px solid #333;border-radius:4px;max-height:240px}"
        ".sum{color:#aaa;font-size:14px}.warn{color:#fc6}a{color:#9cf}</style>",
        "<h1>Dataset review — boxes drawn under unified class mapping</h1>",
        "<p class=sum>Red boxes show the labels for the named class after remapping "
        f"to the COCO-80 + package scheme. Up to {N} images per class per dataset.</p>",
    ]
    # table of contents
    parts.append("<p class=sum>Datasets: " +
                 " · ".join(f"<a href='#{slug(l)}'>{html.escape(l)}</a>"
                            for l, _, _ in entries) + "</p>")

    for label, source, units in entries:
        print(f"[report] sampling {label} ...")
        buckets = sample(units, source)
        present = sorted(buckets)
        summary = ", ".join(f"{NAMES[u]}({len(buckets[u])})" for u in present) or "—"
        parts.append(f"<h2 id='{slug(label)}'>{html.escape(label)}</h2>")
        parts.append(f"<p class=sum>classes found: {html.escape(summary)}</p>")
        for u in present:
            samples = buckets[u]
            note = "" if len(samples) >= N else \
                f" <span class=warn>(only {len(samples)} found)</span>"
            parts.append(f"<h3>{html.escape(NAMES[u])}{note}</h3><div class=row>")
            for i, (img, boxes) in enumerate(samples):
                rel = f"imgs/{slug(label)}__{slug(NAMES[u])}__{i}.jpg"
                if render(img, boxes, OUT / rel):
                    parts.append(f"<img src='{rel}' title='{html.escape(str(img))}'>")
            parts.append("</div>")

    (OUT / "index.html").write_text("\n".join(parts))
    print(f"[report] wrote {OUT/'index.html'}")


if __name__ == "__main__":
    main()
