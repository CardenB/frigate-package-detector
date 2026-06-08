#!/usr/bin/env python3
"""Evaluate a model and log a versioned, per-run benchmark with examples.

Writes report/benchmarks/<schema>/<name>__<timestamp>/ containing metrics
(manifest.json, per_class.json), top failure/positive example images per class,
and a gallery.html to browse them. Runs accumulate so make_benchmark_report.py
can compare models side by side and you can reference any run later.

Default = stock COCO-pretrained yolov9s on the pulled COCO val split — the
catastrophic-forgetting reference.

    python scripts/eval_baseline.py                                  # stock, all classes
    python scripts/eval_baseline.py --classes package,person --examples 20
    python scripts/eval_baseline.py --model models/yolov9s-package/weights/best.pt \
                                    --name yolov9s-package
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

import bench
from common import RAW, classes_cfg, train_cfg
from ultralytics import YOLO

DEFAULT_DATA = RAW / "coco" / "validation" / "dataset.yaml"
# BGR colors for cv2
COLORS = {"positives": (0, 170, 0), "false_positives": (0, 0, 220),
          "false_negatives": (0, 140, 255)}


def val_dirs(data_yaml: Path) -> tuple[Path, Path]:
    d = yaml.safe_load(data_yaml.read_text())
    base = Path(d.get("path", data_yaml.parent))
    img_dir = (base / d["val"]).resolve()
    if not img_dir.is_dir():
        img_dir = img_dir.parent
    lbl_dir = Path(str(img_dir).replace("/images", "/labels"))
    return img_dir, lbl_dir


def ensure_eval_yaml(src: Path, run: Path, split: str = "val") -> Path:
    """Write a patched data.yaml for Ultralytics val(): evaluate `split` as the
    val set (so --split test runs the held-out test split) and add the `train:`
    key FiftyOne omits."""
    d = yaml.safe_load(src.read_text())
    if split != "val" and split in d:
        d["val"] = d[split]
    d.setdefault("train", d.get("val"))
    out = run / "_eval_data.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False))
    return out


def save_confusion_matrix(metrics, run: Path, names: dict):
    """Persist the confusion matrix (raw json + normalized & raw PNGs).

    Ultralytics attaches it to the returned metrics object (model.validator is
    None after val()). Rendered ourselves since we pass plots=False. Returns a
    manifest sub-dict (relative paths) or None.
    """
    cm = getattr(metrics, "confusion_matrix", None)
    if cm is None or getattr(cm, "matrix", None) is None:
        return None
    try:
        import numpy as np
        names_list = [names[i] for i in sorted(names)]
        (run / "confusion_matrix.json").write_text(json.dumps({
            "labels": names_list + ["background"],
            "axes": "rows=predicted, cols=true (last=background)",
            "matrix": np.asarray(cm.matrix).tolist(),
        }))
        cm.plot(normalize=True, save_dir=str(run))   # confusion_matrix_normalized.png
        cm.plot(normalize=False, save_dir=str(run))  # confusion_matrix.png
        return {"json": "confusion_matrix.json", "png": "confusion_matrix.png",
                "png_normalized": "confusion_matrix_normalized.png"}
    except Exception as e:  # noqa: BLE001
        print(f"[eval] confusion matrix save failed: {e}")
        return None


def count_support(lbl_dir: Path) -> tuple[Counter, int]:
    support, n = Counter(), 0
    for txt in lbl_dir.rglob("*.txt"):
        n += 1
        for line in txt.read_text().splitlines():
            if line.strip():
                support[int(line.split()[0])] += 1
    return support, n


def mine_examples(model, img_dir, lbl_dir, names, imgsz, conf, iou_thr,
                  topk, keep_ids, run: Path):
    """Run predictions, match to GT, render top-K TP/FP/FN per class.

    Returns {cls_id: {"counts": {...}, "positives": [...], ...}} with example
    paths relative to the run dir.
    """
    # candidates[cls][category] = list of dicts to rank then render
    cand = defaultdict(lambda: {c: [] for c in bench.CATEGORIES})
    counts = defaultdict(lambda: Counter())

    for r in model.predict(source=str(img_dir), stream=True, conf=conf,
                           imgsz=imgsz, verbose=False):
        h, w = r.orig_shape
        img_path = Path(r.path)
        gts = bench.load_gt(lbl_dir / f"{img_path.stem}.txt", w, h)
        b = r.boxes
        preds = [(int(b.cls[i]), float(b.conf[i]),
                  tuple(float(x) for x in b.xyxy[i]))
                 for i in range(len(b))]
        tps, fps, fns = bench.match(preds, gts, iou_thr)

        for pi, gj, _ in tps:
            cls, cf, box = preds[pi]
            counts[cls]["tp"] += 1
            if keep_ids is None or cls in keep_ids:
                cand[cls]["positives"].append(
                    {"path": img_path, "box": box, "conf": cf, "stem": img_path.stem})
        for pi in fps:
            cls, cf, box = preds[pi]
            counts[cls]["fp"] += 1
            if keep_ids is None or cls in keep_ids:
                cand[cls]["false_positives"].append(
                    {"path": img_path, "box": box, "conf": cf, "stem": img_path.stem})
        for gj in fns:
            cls, box = gts[gj]
            counts[cls]["fn"] += 1
            area = (box[2] - box[0]) * (box[3] - box[1])
            if keep_ids is None or cls in keep_ids:
                cand[cls]["false_negatives"].append(
                    {"path": img_path, "box": box, "area": area, "stem": img_path.stem})

    result = {}
    for cls, cats in cand.items():
        name = names[cls]
        cdir = run / "classes" / bench.safe_class_dir(name)
        entry = {"counts": dict(counts[cls]), "positives": [],
                 "false_positives": [], "false_negatives": []}
        for cat in bench.CATEGORIES:
            items = cats[cat]
            if cat == "false_negatives":
                items.sort(key=lambda d: d["area"], reverse=True)
            else:
                items.sort(key=lambda d: d["conf"], reverse=True)
            for rank, it in enumerate(items[:topk]):
                if "conf" in it:
                    label = f"{name} {it['conf']:.2f}"
                    fname = f"{rank:02d}_conf{it['conf']:.2f}_{it['stem']}.jpg"
                else:
                    label = f"{name} MISSED"
                    fname = f"{rank:02d}_{it['stem']}.jpg"
                out = cdir / cat / fname
                if bench.draw_and_save(it["path"], it["box"], label,
                                       COLORS[cat], out):
                    entry[cat].append(str(out.relative_to(run)))
        result[cls] = entry
    return result


def write_gallery(run: Path, manifest: dict, per_class: list) -> None:
    rows = []
    for c in per_class:
        ex = c.get("examples")
        if not ex or not any(ex[cat] for cat in bench.CATEGORIES):
            continue
        cnt = ex["counts"]
        cat_key = {"positives": "tp", "false_positives": "fp",
                   "false_negatives": "fn"}
        cols = []
        for cat, title in (("positives", "Top positives (TP)"),
                           ("false_positives", "Top false positives"),
                           ("false_negatives", "Top false negatives")):
            imgs = "".join(f'<img src="{p}" loading="lazy">' for p in ex[cat])
            n = cnt.get(cat_key[cat], 0)
            cols.append(f'<div class=col><h4>{title} '
                        f'<span class=muted>(total {n})</span></h4>'
                        f'{imgs or "<i class=muted>none</i>"}</div>')
        rows.append(f'<section id="{bench.safe_class_dir(c["name"])}">'
                    f'<h3>{c["name"]} <span class=muted>'
                    f'(support {c["support"]}, TP {cnt.get("tp",0)} / '
                    f'FP {cnt.get("fp",0)} / FN {cnt.get("fn",0)})</span></h3>'
                    f'<div class=cols>{"".join(cols)}</div></section>')
    nav = " · ".join(
        f'<a href="#{bench.safe_class_dir(c["name"])}">{c["name"]}</a>'
        for c in per_class if c.get("examples")
        and any(c["examples"][cat] for cat in bench.CATEGORIES))
    html = f"""<!doctype html><meta charset=utf-8><title>{manifest['name']} gallery</title>
<style>body{{font:14px system-ui;margin:20px}}.cols{{display:flex;gap:16px}}
.col{{flex:1}}img{{width:100%;max-width:260px;margin:3px 0;border:1px solid #ccc}}
.muted{{color:#888;font-weight:400}}h3{{margin-top:28px;border-top:1px solid #eee;padding-top:12px}}
nav{{position:sticky;top:0;background:#fff;padding:8px 0;font-size:12px;line-height:1.8}}</style>
<h1>{manifest['name']} — examples</h1>
<div class=muted>{manifest['model']} · imgsz {manifest['imgsz']} · conf {manifest['examples']['conf']} · IoU {manifest['examples']['iou']} · {manifest['timestamp']}</div>
<nav>{nav}</nav>{''.join(rows)}"""
    (run / "gallery.html").write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov9s.pt")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--imgsz", type=int, default=train_cfg().get("imgsz", 640))
    ap.add_argument("--name", default="yolov9s-stock")
    ap.add_argument("--examples", type=int, default=12,
                    help="top-K example images per class/category (0 to skip)")
    ap.add_argument("--classes", default=None,
                    help="comma list to mine (default: security_classes in classes.yaml)")
    ap.add_argument("--all-classes", action="store_true",
                    help="mine examples for all 80 classes (override the security set)")
    ap.add_argument("--conf", type=float, default=0.25, help="example conf threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="TP IoU threshold")
    ap.add_argument("--split", default="val",
                    help="which split key in the data.yaml to evaluate (e.g. test)")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = bench.run_dir(args.name, ts)
    run.mkdir(parents=True, exist_ok=True)

    data_yaml = ensure_eval_yaml(Path(args.data), run, args.split)
    img_dir, lbl_dir = val_dirs(data_yaml)
    print(f"[eval] model={args.model} data={data_yaml} imgsz={args.imgsz}")

    model = YOLO(args.model)
    n_params = sum(p.numel() for p in model.model.parameters())
    # plots=True is REQUIRED: Ultralytics only accumulates the confusion matrix
    # when plots is on (val.py gates confusion_matrix.process_batch on it).
    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz,
                        project=str(run), name="_val", verbose=False, plots=True,
                        exist_ok=True)
    support, n_images = count_support(lbl_dir)
    names = metrics.names
    box = metrics.box
    cm_info = save_confusion_matrix(metrics, run, names)

    per_class = []
    metric_by_id = {}
    for idx, cls_id in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(idx)
        cid = int(cls_id)
        metric_by_id[cid] = {
            "id": cid, "name": names[cid], "support": int(support.get(cid, 0)),
            "precision": round(float(p), 4), "recall": round(float(r), 4),
            "map50": round(float(ap50), 4), "map50_95": round(float(ap), 4)}

    # Example mining
    examples_by_id = {}
    if args.examples > 0:
        if args.all_classes:
            keep_ids = None
            print("[eval] mining examples for ALL classes")
        else:
            if args.classes:
                want = {c.strip().lower() for c in args.classes.split(",")}
            else:
                want = {c.strip().lower()
                        for c in classes_cfg().get("security_classes", [])}
            keep_ids = {i for i, n in names.items() if n.lower() in want}
            print(f"[eval] mining examples for security classes: "
                  f"{sorted(names[i] for i in keep_ids)}")
        examples_by_id = mine_examples(model, img_dir, lbl_dir, names, args.imgsz,
                                       args.conf, args.iou, args.examples,
                                       keep_ids, run)

    # Assemble per_class (metrics + examples), ordered by support desc
    all_ids = sorted(set(metric_by_id) | set(examples_by_id),
                     key=lambda i: -int(support.get(i, 0)))
    for cid in all_ids:
        entry = metric_by_id.get(cid, {
            "id": cid, "name": names[cid], "support": int(support.get(cid, 0)),
            "precision": None, "recall": None, "map50": None, "map50_95": None})
        if cid in examples_by_id:
            entry["examples"] = examples_by_id[cid]
        per_class.append(entry)

    manifest = {
        "schema_version": bench.SCHEMA_VERSION,
        "name": args.name, "model": args.model, "model_params": int(n_params),
        "dataset": str(args.data), "imgsz": args.imgsz, "timestamp": ts,
        "num_images": n_images, "num_instances": int(sum(support.values())),
        "overall": {"map50_95": round(float(box.map), 4),
                    "map50": round(float(box.map50), 4),
                    "precision": round(float(box.mp), 4),
                    "recall": round(float(box.mr), 4)},
        "split": args.split,
        "examples": {"enabled": args.examples > 0, "k": args.examples,
                     "conf": args.conf, "iou": args.iou},
        "confusion_matrix": cm_info,
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run / "per_class.json").write_text(json.dumps(per_class, indent=2) + "\n")
    if args.examples > 0:
        write_gallery(run, manifest, per_class)

    o = manifest["overall"]
    print(f"\n[eval] === {args.name} ({run.name}) ===")
    print(f"[eval] mAP50-95 {o['map50_95']:.4f}  mAP50 {o['map50']:.4f}  "
          f"P {o['precision']:.4f}  R {o['recall']:.4f}")
    print(f"[eval] run dir -> {run}")
    if args.examples > 0:
        print(f"[eval] gallery -> {run/'gallery.html'}")
    print("[eval] rebuild side-by-side: python scripts/make_benchmark_report.py")


if __name__ == "__main__":
    main()
