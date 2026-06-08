"""Shared benchmark layout + error-analysis helpers.

Layout (schema-version first, then one folder per run):

    report/benchmarks/
      v1/
        <name>__<YYYYMMDD-HHMMSS>/
          manifest.json        # overall metrics + metadata + schema_version
          per_class.json       # per-class metrics, support, tp/fp/fn counts + paths
          gallery.html         # browse example images by class & category
          classes/<cls>/
            positives/         # top-K correct, high-confidence detections (TP)
            false_positives/   # top-K confident wrong detections (FP)
            false_negatives/   # top-K missed ground-truth instances (FN)

Bump SCHEMA_VERSION when manifest/per_class shape changes so old runs stay
comparable under their own version folder.
"""
from __future__ import annotations

from pathlib import Path

from common import REPO

SCHEMA_VERSION = "v1"
CATEGORIES = ("positives", "false_positives", "false_negatives")


def bench_root() -> Path:
    return REPO / "report" / "benchmarks"


def schema_dir() -> Path:
    return bench_root() / SCHEMA_VERSION


def run_dir(name: str, timestamp: str) -> Path:
    """report/benchmarks/<schema>/<name>__<timestamp>/"""
    return schema_dir() / f"{name}__{timestamp}"


def safe_class_dir(name: str) -> str:
    """Filesystem-safe class folder name ('traffic light' -> 'traffic_light')."""
    return name.replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Geometry + GT loading + pred/GT matching
# ---------------------------------------------------------------------------
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(label_path: Path, w: int, h: int):
    """YOLO label file -> [(cls_id, (x1,y1,x2,y2) px)]."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        cls = int(float(t[0]))
        cx, cy, bw, bh = (float(x) for x in t[1:5])
        out.append((cls, (
            (cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def match(preds, gts, iou_thr=0.5):
    """Greedy match by confidence. preds=[(cls,conf,xyxy)], gts=[(cls,xyxy)].

    Returns (tps, fps, fns):
      tps = [(pred_idx, gt_idx, iou)]   fps = [pred_idx]   fns = [gt_idx]
    """
    order = sorted(range(len(preds)), key=lambda i: preds[i][1], reverse=True)
    gt_used = [False] * len(gts)
    tps, fps = [], []
    for pi in order:
        pcls, _, pbox = preds[pi]
        best_j, best_iou = -1, iou_thr
        for j, (gcls, gbox) in enumerate(gts):
            if gt_used[j] or gcls != pcls:
                continue
            v = iou(pbox, gbox)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            gt_used[best_j] = True
            tps.append((pi, best_j, best_iou))
        else:
            fps.append(pi)
    fns = [j for j, used in enumerate(gt_used) if not used]
    return tps, fps, fns


def draw_and_save(img_path: Path, box, label: str, color, out_path: Path,
                  max_w: int = 720) -> bool:
    """Draw one box+label on the image, downscale, save JPG. Returns success."""
    import cv2  # local import so non-image code paths don't need cv2

    img = cv2.imread(str(img_path))
    if img is None:
        return False
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, max(8, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 72]))
