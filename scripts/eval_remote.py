#!/usr/bin/env python3
"""Benchmark a detector on a fixed vocabulary + val set, scored with COCO mAP.

Built to benchmark a remote open-vocabulary "locate-anything" service against the
YOLO finetune on the SAME data, vocabulary, and scorer.

Backends:
  --detector remote   POST images to a LAN FastAPI service (open-vocab detector)
  --detector yolo     a local ultralytics .pt/.onnx (apples-to-apples reference)

Both are scored identically with pycocotools and logged to
report/benchmarks/<name>.json in the SAME schema as eval_baseline.py, so
make_benchmark_report.py puts them side by side.

Default val set = data/final (our merged val — it has `package` GT, which COCO
val does NOT). Default vocab = the security-relevant subset.

    # see the raw response shape of your service, then we lock the parser:
    python scripts/eval_remote.py --detector remote --probe
    # full run:
    python scripts/eval_remote.py --detector remote --name locate-anything
    # reference (same set+scorer) for the YOLO finetune:
    python scripts/eval_remote.py --detector yolo \
        --model models/yolov9s-package/weights/best.pt --name yolov9s-package-coco
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image

import bench
from common import CONFIGS, FINAL, REPO, classes_cfg

BENCH_DIR = REPO / "report" / "benchmarks"
DEFAULT_VOCAB = ["person", "car", "dog", "cat", "package"]


def load_scenario(name: str) -> dict:
    """A named (dataset, vocabulary) preset from configs/benchmarks.yaml."""
    cfg = yaml.safe_load((CONFIGS / "benchmarks.yaml").read_text())
    scenarios = cfg.get("scenarios", {})
    if name not in scenarios:
        sys.exit(f"scenario '{name}' not in configs/benchmarks.yaml "
                 f"(have: {', '.join(scenarios)})")
    return scenarios[name]


def expand_vocab(spec) -> list[str]:
    """vocab spec -> class-name list. Accepts a list, a comma string, or the
    sentinel 'coco80' (all 80 COCO classes — classes.yaml minus `package`)."""
    if isinstance(spec, list):
        return [s.strip() for s in spec if str(s).strip()]
    if spec == "coco80":
        return [c for c in classes_cfg()["classes"] if c != "package"]
    return [s.strip() for s in str(spec).split(",") if s.strip()]

# ---------------------------------------------------------------------------
# Remote "locate-anything" (LocateAnything VLM) contract. JSON body to POST
# /locate; vocabulary is a prompt string; response boxes are pixel xyxy in the
# server's (downscaled) image_size with NO confidence. All service-specific
# behavior is confined to these two functions + the env URL.
# ---------------------------------------------------------------------------
REMOTE_URL = os.environ.get("REMOTE_DETECTOR_URL", "")  # e.g. http://10.0.0.10:8002
REMOTE_ROUTE = os.environ.get("REMOTE_DETECTOR_ROUTE", "/locate")
REMOTE_BATCH_ROUTE = os.environ.get("REMOTE_DETECTOR_BATCH_ROUTE", "/locate_batch")
REMOTE_MODE = os.environ.get("REMOTE_DETECTOR_MODE", "hybrid")  # fast|slow|hybrid
# A pathological image can generate toward max_new_tokens (~40-70s at int8), so
# give it room rather than clipping it to a "failure" that trips the breaker.
REMOTE_TIMEOUT = float(os.environ.get("REMOTE_DETECTOR_TIMEOUT", "90"))
# Server caps the longest side to this before inference (LA_MAX_SIDE, /health
# reports it). Used to reconstruct the downscaled size when a batch result omits
# image_size, so box rescaling stays correct.
REMOTE_MAX_SIDE = int(os.environ.get("REMOTE_DETECTOR_MAX_SIDE", "640"))
# The server queues requests (max ~8 in flight) and returns 503 when full. Per
# the service contract that's "back off and retry", not an error — so we retry a
# 503 a few times with growing backoff instead of counting it as a failure.
REMOTE_503_RETRIES = int(os.environ.get("REMOTE_DETECTOR_503_RETRIES", "6"))
# Abort after this many CONSECUTIVE failures (any success resets it). Kept small
# so a timeout cascade (stuck server, ~90s each) caps wasted time at ~12min, and
# a down service (ConnectionError, ~ms each) aborts in under a second. A real
# slow-but-completing image is a SUCCESS and never counts here.
CIRCUIT_BREAK = int(os.environ.get("REMOTE_DETECTOR_CIRCUIT_BREAK", "8"))
# LocateAnything emits no per-box confidence; we assign this constant so the
# scorer has a score field. With one score, mAP degenerates to a P/R operating
# point — precision/recall/F1 are the meaningful metrics for this detector.
REMOTE_CONST_SCORE = 1.0

PROMPT_TEMPLATE = "Locate all the instances that matches the following description: {cats}."


def _b64(img_path: Path) -> str:
    import base64
    return base64.b64encode(img_path.read_bytes()).decode("ascii")


def _prompt(vocab: list[str]) -> str:
    return PROMPT_TEMPLATE.format(cats="</c>".join(vocab))


def downscaled_size(w: int, h: int) -> tuple[int, int]:
    """The (W,H) the server resizes to (longest side -> REMOTE_MAX_SIDE)."""
    m = max(w, h)
    if m <= REMOTE_MAX_SIDE:
        return w, h
    s = REMOTE_MAX_SIDE / m
    return round(w * s), round(h * s)


def remote_build_request(img_path: Path, vocab: list[str], conf: float) -> dict:
    """JSON body for POST /locate. Image is base64; vocab is a </c>-joined prompt."""
    return {"image": _b64(img_path), "prompt": _prompt(vocab), "mode": REMOTE_MODE}


def remote_parse_response(payload: dict, w: int, h: int):
    """Map a /locate (or one /locate_batch result) -> [(label, score, xyxy px)].

    Box coords are pixels in `image_size` (the DOWNSCALED frame). Batch results
    may omit image_size, so fall back to the size we'd expect from REMOTE_MAX_SIDE.
    Either way we rescale back to the original (w, h) the GT is in.
    """
    rw, rh = payload.get("image_size") or downscaled_size(w, h)
    sx = w / rw if rw else 1.0
    sy = h / rh if rh else 1.0
    out = []
    for b in payload.get("boxes", []):
        label = b.get("label")
        if label is None:          # bare box with no <ref> tag — unassignable
            continue
        try:
            x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError):
            continue
        out.append((label, REMOTE_CONST_SCORE,
                    [x1 * sx, y1 * sy, x2 * sx, y2 * sy]))
    return out


# ---------------------------------------------------------------------------
# Detectors: predict(img_path, w, h) -> [(label, score, [x1,y1,x2,y2] pixels)]
# ---------------------------------------------------------------------------
class RemoteDetector:
    def __init__(self, vocab, conf):
        import requests  # lazy
        self._requests = requests
        self.vocab, self.conf = vocab, conf
        if not REMOTE_URL:
            sys.exit("REMOTE_DETECTOR_URL not set (.env). e.g. http://<host>:<port>")
        base = REMOTE_URL.rstrip("/")
        self.url = base + REMOTE_ROUTE
        self.batch_url = base + REMOTE_BATCH_ROUTE
        self.last_timing: dict = {}   # server-reported ms from the most recent call

    def _post(self, url: str, body: dict, timeout: float) -> dict:
        """POST with 503 back-off-and-retry — 503 = server queue full (transient),
        not a failure. Other errors/timeouts propagate to the caller's handler."""
        delay = 2.0
        for attempt in range(REMOTE_503_RETRIES):
            r = self._requests.post(url, json=body, timeout=timeout)
            if r.status_code == 503 and attempt < REMOTE_503_RETRIES - 1:
                time.sleep(delay); delay = min(delay * 2, 15)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()   # exhausted retries on a persistent 503
        return r.json()

    def raw(self, img_path: Path) -> dict:
        body = remote_build_request(img_path, self.vocab, self.conf)
        return self._post(self.url, body, REMOTE_TIMEOUT)

    def predict(self, img_path: Path, w: int, h: int):
        payload = self.raw(img_path)
        self.last_timing = {"inference_ms": payload.get("inference_ms"),
                            "queue_wait_ms": payload.get("queue_wait_ms"),
                            "server_ms": payload.get("latency_ms")}
        return remote_parse_response(payload, w, h)

    def predict_batch(self, items):
        """items = [(img_path, w, h)]; one /locate_batch forward for all of them.

        Returns a list aligned to items: each is [(label, score, xyxy px)].
        The GPU is single-tenant, so batching (not concurrency) is the real
        throughput win. Returns [] for any image the server didn't answer.
        """
        body = {"images": [_b64(p) for p, _, _ in items],
                "prompt": _prompt(self.vocab), "mode": REMOTE_MODE}
        timeout = max(REMOTE_TIMEOUT, 8.0 * len(items))
        payload = self._post(self.batch_url, body, timeout)
        self.last_timing = {"server_ms": payload.get("latency_ms"),
                            "per_image_ms": payload.get("per_image_ms")}
        results = payload.get("results", [])
        out = [remote_parse_response(res, w, h)
               for (_, w, h), res in zip(items, results)]
        out += [[]] * (len(items) - len(out))   # pad if server returned fewer
        return out


class YoloDetector:
    """Local ultralytics model scored through the SAME pipeline for comparison."""

    def __init__(self, model_path, vocab, conf, imgsz):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.names = self.model.names
        self.vocab = set(vocab)
        self.conf, self.imgsz = conf, imgsz

    def predict(self, img_path: Path, w: int, h: int):
        res = self.model.predict(str(img_path), imgsz=self.imgsz, conf=self.conf,
                                 verbose=False, device=0)[0]
        out = []
        for b in res.boxes:
            label = self.names[int(b.cls)]
            if label not in self.vocab:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append((label, float(b.conf), [x1, y1, x2, y2]))
        return out


# ---------------------------------------------------------------------------
# Val set -> COCO ground truth (restricted to the vocab)
# ---------------------------------------------------------------------------
def load_val(data_yaml: Path, vocab: list[str], limit: int | None,
             split: str = "val", only_with_gt: bool = False):
    d = yaml.safe_load(data_yaml.read_text())
    names = d["names"]
    id2name = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
    name2id = {n: i for i, n in id2name.items()}
    # unified-id -> vocab category id (1-based for COCO); drop non-vocab
    cat_of_uid = {name2id[n]: vi + 1 for vi, n in enumerate(vocab) if n in name2id}
    missing = [n for n in vocab if n not in name2id]
    if missing:
        print(f"[warn] vocab not in dataset names, will have 0 support: {missing}")

    base = Path(d.get("path", data_yaml.parent))
    split_rel = d.get(split)
    if split_rel is None:
        sys.exit(f"split '{split}' not in {data_yaml} (have: "
                 f"{[k for k in ('train','val','test') if k in d]})")
    val_img_dir = (base / split_rel).resolve()
    img_files = sorted(p for p in val_img_dir.rglob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    # Shuffle (seeded) so sources interleave — sorted order clusters by source
    # (all coco, then all package), which means a partial run (or --limit) can
    # miss an entire class. Shuffling makes any prefix a representative sample.
    random.Random(0).shuffle(img_files)
    if limit:
        img_files = img_files[:limit]

    images, annotations = [], []
    ann_id = img_id = skipped = 0
    for img in img_files:
        try:
            with Image.open(img) as im:
                w, h = im.size
        except (FileNotFoundError, OSError):
            skipped += 1   # dangling symlink / corrupt file — don't kill the run
            continue
        img_id += 1
        images.append({"id": img_id, "file": img, "width": w, "height": h})
        lbl = Path(str(img.parent).replace("/images", "/labels")) / (img.stem + ".txt")
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            uid = int(float(t[0]))
            if uid not in cat_of_uid:
                continue
            cx, cy, bw, bh = (float(x) for x in t[1:5])
            x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cat_of_uid[uid],
                "bbox": [x, y, bw * w, bh * h], "area": bw * w * bh * h, "iscrowd": 0,
            })
            ann_id += 1

    if skipped:
        print(f"[warn] skipped {skipped} unreadable image(s) (dangling symlink/corrupt)")

    if only_with_gt:
        # Focus on frames that actually contain a vocab object (e.g. for a
        # package eval on a VLM that's slow per image). NOTE: this drops pure
        # backgrounds, so precision excludes background false positives.
        with_gt = {a["image_id"] for a in annotations}
        images = [im for im in images if im["id"] in with_gt]

    categories = [{"id": vi + 1, "name": n} for vi, n in enumerate(vocab)]
    gt = {"images": [{"id": im["id"], "width": im["width"], "height": im["height"]}
                     for im in images],
          "annotations": annotations, "categories": categories}
    return images, gt, {n: vi + 1 for vi, n in enumerate(vocab)}


def score_coco(gt: dict, detections: list, vocab: list[str]):
    """pycocotools eval -> overall + per-class (map, map50, best-F1 P/R, support)."""
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    zeros = [{"id": k + 1, "name": n, "support": 0, "precision": 0.0,
              "recall": 0.0, "map50": 0.0, "map50_95": 0.0}
             for k, n in enumerate(vocab)]
    if not gt["images"] or not gt["annotations"]:
        # nothing to score against (e.g. an all-skipped slice) — don't crash
        return {"map50_95": 0.0, "map50": 0.0, "precision": 0.0, "recall": 0.0}, zeros

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])

    E = COCOeval(coco_gt, coco_dt, "bbox")
    E.evaluate(); E.accumulate()
    prec = E.eval["precision"]            # [T, R, K, A, M]
    rec_thrs = E.params.recThrs

    support = {c["id"]: 0 for c in gt["categories"]}
    for a in gt["annotations"]:
        support[a["category_id"]] += 1

    classes, present = [], []
    for k, name in enumerate(vocab):
        cat_id = k + 1
        p_all = prec[:, :, k, 0, -1]
        ap = float(p_all[p_all > -1].mean()) if (p_all > -1).any() else 0.0
        p50 = prec[0, :, k, 0, -1]
        valid = p50 > -1
        ap50 = float(p50[valid].mean()) if valid.any() else 0.0
        if valid.any():
            pp, rc = p50[valid], rec_thrs[valid]
            f1 = 2 * pp * rc / (pp + rc + 1e-9)
            bi = int(f1.argmax())
            precision, recall = float(pp[bi]), float(rc[bi])
        else:
            precision = recall = 0.0
        sup = support.get(cat_id, 0)
        classes.append({"id": cat_id, "name": name, "support": sup,
                        "precision": round(precision, 4), "recall": round(recall, 4),
                        "map50": round(ap50, 4), "map50_95": round(ap, 4)})
        if sup > 0:
            present.append(classes[-1])

    def mean(key):
        return round(float(np.mean([c[key] for c in present])), 4) if present else 0.0

    overall = {"map50_95": mean("map50_95"), "map50": mean("map50"),
               "precision": mean("precision"), "recall": mean("recall")}
    return overall, classes


def summarize_latency(latencies: list) -> dict:
    """mean/p50/p95/max for each timing field present (wall + server-reported)."""
    import statistics
    out = {"n": len(latencies)}
    for key in ("wall_ms", "server_ms", "inference_ms", "queue_wait_ms"):
        vals = sorted(r[key] for r in latencies if r.get(key) is not None)
        if not vals:
            continue
        def pct(q):
            return round(vals[min(len(vals) - 1, int(q * len(vals)))], 1)
        out[key] = {"mean": round(statistics.fmean(vals), 1),
                    "p50": pct(0.50), "p95": pct(0.95), "max": round(vals[-1], 1)}
    return out


# BGR colors for cv2 (match eval_baseline's gallery).
COLORS = {"positives": (0, 170, 0), "false_positives": (0, 0, 220),
          "false_negatives": (0, 140, 255)}


def mine_examples(images, preds_by_img, gts_by_img, vocab, keep_idx, topk, run):
    """Match preds vs GT per image, render top-K TP/FP/FN per class.

    preds_by_img[id] = [(vocab_idx, score, xyxy)];  gts_by_img[id] = [(vocab_idx, xyxy)].
    Returns {vocab_idx: {counts, positives[], false_positives[], false_negatives[]}}.
    (Remote has a constant score, so positives/FPs aren't conf-ranked — first K.)
    """
    cand = defaultdict(lambda: {c: [] for c in bench.CATEGORIES})
    counts = defaultdict(lambda: defaultdict(int))
    for im in images:
        preds = preds_by_img.get(im["id"], [])
        gts = gts_by_img.get(im["id"], [])
        tps, fps, fns = bench.match(preds, gts, 0.5)
        path, stem = im["file"], im["file"].stem
        for pi, gj, _ in tps:
            cls, cf, box = preds[pi]; counts[cls]["tp"] += 1
            if keep_idx is None or cls in keep_idx:
                cand[cls]["positives"].append({"path": path, "box": box, "conf": cf, "stem": stem})
        for pi in fps:
            cls, cf, box = preds[pi]; counts[cls]["fp"] += 1
            if keep_idx is None or cls in keep_idx:
                cand[cls]["false_positives"].append({"path": path, "box": box, "conf": cf, "stem": stem})
        for gj in fns:
            cls, box = gts[gj]; counts[cls]["fn"] += 1
            if keep_idx is None or cls in keep_idx:
                area = (box[2] - box[0]) * (box[3] - box[1])
                cand[cls]["false_negatives"].append({"path": path, "box": box, "area": area, "stem": stem})

    result = {}
    for cls, cats in cand.items():
        name = vocab[cls]
        cdir = run / "classes" / bench.safe_class_dir(name)
        entry = {"counts": dict(counts[cls]), "positives": [],
                 "false_positives": [], "false_negatives": []}
        for cat in bench.CATEGORIES:
            items = cats[cat]
            items.sort(key=lambda d: d.get("area", d.get("conf", 0)), reverse=True)
            for rank, it in enumerate(items[:topk]):
                label = f"{name} MISSED" if cat == "false_negatives" else f"{name} {it['conf']:.2f}"
                out = cdir / cat / f"{rank:02d}_{it['stem']}.jpg"
                if bench.draw_and_save(it["path"], it["box"], label, COLORS[cat], out):
                    entry[cat].append(str(out.relative_to(run)))
        result[cls] = entry
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None,
                    help="preset (data+vocab) from configs/benchmarks.yaml, e.g. coco|package")
    ap.add_argument("--detector", choices=["remote", "yolo"], default="remote")
    ap.add_argument("--model", default="yolov9s.pt", help="for --detector yolo")
    ap.add_argument("--data", default=None, help="override scenario dataset (data.yaml)")
    ap.add_argument("--classes", default=None,
                    help="override vocab: comma list or 'coco80'")
    ap.add_argument("--split", default=None, help="val|test (override scenario)")
    ap.add_argument("--only-with-gt", dest="only_with_gt", action="store_true",
                    default=None, help="keep only frames with a vocab GT box")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="low conf for mAP (sweep needs full PR curve)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=1,
                    help="(ignored for remote — the service is single-GPU/no-queue, "
                         "so we run serial. Use --batch for throughput.)")
    ap.add_argument("--batch", type=int, default=1,
                    help="remote: images per /locate_batch forward (the real throughput "
                         "lever on a single-GPU service). 1 = per-image /locate.")
    ap.add_argument("--limit", type=int, default=None, help="first N images (smoke)")
    ap.add_argument("--name", default=None, help="benchmark record label")
    ap.add_argument("--examples", type=int, default=12,
                    help="top-K TP/FP/FN example images per vocab class (0 to skip gallery)")
    ap.add_argument("--no-wandb", dest="no_wandb", action="store_true",
                    help="don't log this run to the self-hosted wandb")
    ap.add_argument("--probe", action="store_true",
                    help="remote: POST one image, print raw JSON, exit")
    args = ap.parse_args()

    # Resolve from scenario, with explicit CLI flags taking precedence.
    sc = load_scenario(args.scenario) if args.scenario else {}
    data = args.data or sc.get("data") or str(FINAL / "data.yaml")
    split = args.split or sc.get("split") or "val"
    vocab = expand_vocab(args.classes or sc.get("vocab") or DEFAULT_VOCAB)
    only_with_gt = (args.only_with_gt if args.only_with_gt is not None
                    else bool(sc.get("only_with_gt", False)))

    model_tag = "locate-anything" if args.detector == "remote" else Path(args.model).stem
    name = args.name or (f"{args.scenario}__{model_tag}" if args.scenario else model_tag)

    data_yaml = Path(data)
    if not data_yaml.exists():
        sys.exit(f"{data_yaml} missing — run `make build` first.")

    if args.probe:
        images, _, _ = load_val(data_yaml, vocab, limit=1, split=split)
        det = RemoteDetector(vocab, args.conf)
        print(f"[probe] POST {det.url}  classes={vocab}")
        print(json.dumps(det.raw(images[0]["file"]), indent=2))
        return

    print(f"[bench] scenario={args.scenario or 'adhoc'} detector={args.detector} "
          f"split={split} only_with_gt={only_with_gt}")
    print(f"[bench] vocab={vocab}")
    images, gt, _ = load_val(data_yaml, vocab, args.limit, split, only_with_gt)
    print(f"[bench] {len(images)} images, {len(gt['annotations'])} GT boxes")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = bench.run_dir(name, ts)
    run.mkdir(parents=True, exist_ok=True)

    if args.detector == "remote":
        det = RemoteDetector(vocab, args.conf)
    else:
        det = YoloDetector(args.model, vocab, args.conf, args.imgsz)
    name2cat = {n: i + 1 for i, n in enumerate(vocab)}

    # Start the wandb run NOW so it's visible in-flight and streams progress below.
    n_params = (sum(p.numel() for p in det.model.model.parameters())
                if args.detector == "yolo" else 0)
    from bench_wandb import init_run, log_progress, finalize_run
    wb = None if args.no_wandb else init_run({
        "name": name, "run_id": run.name, "detector": args.detector,
        "mode": REMOTE_MODE if args.detector == "remote" else None,
        "scenario": args.scenario or "adhoc", "dataset": str(data), "split": split,
        "imgsz": args.imgsz, "scorer": "pycocotools",
        "model": args.model if args.detector == "yolo" else det.url,
        "model_params": int(n_params),
        "num_images": len(images), "num_instances": len(gt["annotations"]),
    })

    # GT per image (xyxy, vocab-idx) for pred/GT matching in the gallery.
    gts_by_img = defaultdict(list)
    for a in gt["annotations"]:
        x, y, bw, bh = a["bbox"]
        gts_by_img[a["image_id"]].append((a["category_id"] - 1, (x, y, x + bw, y + bh)))

    def to_rows(im, preds):
        rows, ptuples = [], []
        for label, score, (x1, y1, x2, y2) in preds:
            if label not in name2cat:
                continue
            rows.append({"image_id": im["id"], "category_id": name2cat[label],
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
            ptuples.append((name2cat[label] - 1, score, (x1, y1, x2, y2)))
        return rows, ptuples

    detections, preds_by_img = [], {}
    latencies = []   # per-image timing: {image, wall_ms, inference_ms?, queue_wait_ms?}
    n = len(images)
    failures = consec = 0   # transient blips tolerated; a CASCADE aborts
    fail_kinds: dict[str, int] = {}   # exception type -> count (Timeout vs ConnectionError vs ...)

    def record(im, preds):
        rows, pt = to_rows(im, preds)
        detections.extend(rows); preds_by_img[im["id"]] = pt

    def timed(im):
        """Run det.predict, capturing wall + server-reported latency."""
        t0 = time.time()
        preds = det.predict(im["file"], im["width"], im["height"])
        rec = {"image": im["file"].name, "wall_ms": round((time.time() - t0) * 1000, 1)}
        rec.update({k: v for k, v in getattr(det, "last_timing", {}).items() if v is not None})
        latencies.append(rec)
        return preds

    def note_fail(where, e):
        k = type(e).__name__
        fail_kinds[k] = fail_kinds.get(k, 0) + 1
        if fail_kinds[k] <= 3:           # sample the first few of each kind, with the offender
            print(f"[warn] {where} -> {k}: {str(e)[:160]}")

    def abort_if_down(done):
        if consec >= CIRCUIT_BREAK:
            finalize_run(wb, run, status="aborted")  # mark the live run failed
            shutil.rmtree(run, ignore_errors=True)   # don't leave a partial run
            kinds = dict(fail_kinds)
            sys.exit(f"[abort] {consec} consecutive failures at {done}/{n} images "
                     f"(types: {kinds}). Timeout => service alive but stuck/slow on an "
                     f"image (single-tenant backlog); ConnectionError => process down. "
                     f"Wrote nothing. Re-run after addressing it.")

    if args.detector == "remote" and args.batch > 1:
        for i in range(0, n, args.batch):
            chunk = images[i:i + args.batch]
            try:
                t0 = time.time()
                preds_list = det.predict_batch(
                    [(im["file"], im["width"], im["height"]) for im in chunk])
                consec = 0
                wall = (time.time() - t0) * 1000
                per = det.last_timing.get("per_image_ms") or wall / len(chunk)
                for im in chunk:
                    latencies.append({"image": im["file"].name,
                                      "wall_ms": round(wall / len(chunk), 1),
                                      "inference_ms": round(per, 1) if per else None})
            except Exception as e:                       # noqa: BLE001
                note_fail(f"batch@{i}", e)
                failures += len(chunk); consec += len(chunk)
                preds_list = [[]] * len(chunk)
                abort_if_down(i)
            for im, preds in zip(chunk, preds_list):
                record(im, preds)
            done = min(i + args.batch, n)
            last = latencies[-1] if latencies else {}
            log_progress(wb, {"progress/images_done": done, "progress/failures": failures,
                              **({"latency/inference_ms": last["inference_ms"]}
                                 if last.get("inference_ms") is not None else {})}, step=done)
            print(f"[bench]   {done}/{n} images (batch={args.batch}, {failures} failed {fail_kinds or ''})")
    elif args.detector == "remote":
        # Serial: the service queues/serializes, so concurrency just adds wait.
        for i, im in enumerate(images, 1):
            try:
                preds = timed(im); consec = 0
            except Exception as e:                       # noqa: BLE001
                preds = []; failures += 1; consec += 1
                note_fail(f"img {i} ({im['file'].name})", e)
                abort_if_down(i)
            record(im, preds)
            if i % 50 == 0:
                last = latencies[-1] if latencies else {}
                m = {"progress/images_done": i, "progress/failures": failures}
                for k in ("server_ms", "queue_wait_ms", "inference_ms"):
                    if last.get(k) is not None:
                        m[f"latency/{k}"] = last[k]
                log_progress(wb, m, step=i)
            if i % 250 == 0:
                last = latencies[-1] if latencies else {}
                print(f"[bench]   {i}/{n} images ({failures} failed {fail_kinds or ''}) "
                      f"~{last.get('server_ms', last.get('wall_ms', '?'))}ms/img")
    else:
        for i, im in enumerate(images, 1):
            record(im, timed(im))
            if i % 250 == 0:
                last = latencies[-1] if latencies else {}
                log_progress(wb, {"progress/images_done": i,
                                  "latency/wall_ms": last.get("wall_ms")}, step=i)
                print(f"[bench]   {i}/{n} images")

    if failures:
        print(f"[warn] {failures}/{n} images failed prediction "
              f"(types: {dict(fail_kinds)}; counted as no detections)")
    print(f"[bench] {len(detections)} predictions -> scoring (pycocotools)")
    overall, classes = score_coco(gt, detections, vocab)

    # Example galleries via greedy matching. The vocab IS already the focused
    # set (5 for package, 80 for coco), so mine the whole vocab by default —
    # this is what guarantees the `package` class gets a gallery. For a large
    # (coco-80) vocab, pass --examples 0 to skip the heavy gallery render.
    examples = {}
    if args.examples > 0:
        examples = mine_examples(images, preds_by_img, gts_by_img, vocab,
                                 None, args.examples, run)

    per_class = []
    for k, c in enumerate(classes):
        entry = dict(c)            # id, name, support, precision, recall, map50, map50_95
        if k in examples:
            entry["examples"] = examples[k]
        per_class.append(entry)

    lat_summary = summarize_latency(latencies)
    if latencies:
        (run / "latency.json").write_text(json.dumps(latencies, indent=2) + "\n")

    manifest = {
        "schema_version": bench.SCHEMA_VERSION,
        "name": name, "model": args.model if args.detector == "yolo" else det.url,
        "model_params": int(n_params), "detector": args.detector,
        "mode": REMOTE_MODE if args.detector == "remote" else None,
        "scenario": args.scenario or "adhoc", "dataset": str(data), "split": split,
        "imgsz": args.imgsz, "scorer": "pycocotools", "timestamp": ts,
        "num_images": len(images), "num_instances": len(gt["annotations"]),
        "failures": failures,
        "overall": overall,
        "latency": lat_summary,
        "examples": {"enabled": args.examples > 0, "k": args.examples,
                     "conf": "const-1.0" if args.detector == "remote" else args.conf,
                     "iou": 0.5},
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run / "per_class.json").write_text(json.dumps(per_class, indent=2) + "\n")
    if args.examples > 0:
        from eval_baseline import write_gallery
        write_gallery(run, manifest, per_class)

    try:
        finalize_run(wb, run, status="finished")
    except Exception as e:                               # noqa: BLE001
        print(f"[wandb] finalize failed: {e} (results saved locally)")

    print(f"\n[bench] === {name} ({run.name}) ===")
    print(f"[bench] mAP50-95 {overall['map50_95']:.4f}  mAP50 {overall['map50']:.4f}  "
          f"P {overall['precision']:.4f}  R {overall['recall']:.4f}")
    for c in classes:
        print(f"[bench]   {c['name']:<10} map50_95 {c['map50_95']:.3f}  "
              f"map50 {c['map50']:.3f}  P {c['precision']:.3f}  R {c['recall']:.3f}  "
              f"(support {c['support']})")
    print(f"[bench] run dir -> {run}")
    print("[bench] rebuild side-by-side: python scripts/make_benchmark_report.py")


if __name__ == "__main__":
    main()
