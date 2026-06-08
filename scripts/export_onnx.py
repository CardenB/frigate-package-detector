#!/usr/bin/env python3
"""Export a trained .pt to ONNX for Frigate's `onnx` detector.

Usage:
    python scripts/export_onnx.py [path/to/best.pt]

Defaults to the most recent models/yolov9s-package*/weights/best.pt. Also writes
a labelmap.txt (index -> class name) for the Frigate config.
"""
from __future__ import annotations

import sys
from pathlib import Path

from common import REPO, train_cfg, unified_classes
from ultralytics import YOLO


def latest_best() -> Path | None:
    # Primary location (train.py anchors project to REPO/models); fall back to
    # runs/ in case Ultralytics' default runs dir was used.
    cands = list((REPO / "models").glob("yolov9s-package*/weights/best.pt"))
    cands += list((REPO / "runs").rglob("yolov9s-package*/weights/best.pt"))
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def main() -> None:
    cfg = train_cfg()
    exp = cfg.get("export", {})

    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_best()
    if not weights or not weights.exists():
        raise SystemExit("No weights found. Pass a path or run train.py first.")

    model = YOLO(str(weights))
    print(f"[export] {weights} -> ONNX (imgsz={cfg['imgsz']}, opset={exp.get('opset',16)})")
    onnx_path = model.export(
        format=exp.get("format", "onnx"),
        imgsz=cfg["imgsz"],
        opset=exp.get("opset", 16),
        simplify=exp.get("simplify", True),
        dynamic=exp.get("dynamic", False),
        half=exp.get("half", False),
    )

    # Labelmap for Frigate.
    names = unified_classes()
    labelmap = Path(onnx_path).with_name("labelmap.txt")
    labelmap.write_text("\n".join(names) + "\n")

    print(f"[export] ONNX:     {onnx_path}")
    print(f"[export] labelmap: {labelmap}")
    verify_contract(Path(onnx_path), len(names), cfg["imgsz"])
    print("[export] Copy both into Frigate's model dir and see frigate/config.snippet.yaml")


def _dim(d) -> object:
    """Extract a dimension as int or its symbolic name."""
    return d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")


def verify_contract(onnx_path: Path, nc: int, imgsz: int) -> None:
    """Assert the exported ONNX matches Frigate's yolo-generic drop-in contract.

    Expected (same interface as Frigate's default yolov9 ONNX, just nc classes):
      input:  [1, 3, imgsz, imgsz]   NCHW float
      output: [1, 4+nc, 8400@640]    raw boxes+scores, NO embedded NMS
    A mismatch here means Frigate will mis-parse the model, so we fail loudly.
    """
    import onnx

    model = onnx.load(str(onnx_path))
    g = model.graph
    in_shape = [_dim(d) for d in g.input[0].type.tensor_type.shape.dim]
    outs = {o.name: [_dim(d) for d in o.type.tensor_type.shape.dim] for o in g.output}
    print(f"[verify] input  {g.input[0].name}: {in_shape}")
    for n, s in outs.items():
        print(f"[verify] output {n}: {s}")

    problems = []
    if in_shape[:2] != [1, 3] or in_shape[2:] != [imgsz, imgsz]:
        problems.append(f"input {in_shape} != [1, 3, {imgsz}, {imgsz}] (NCHW)")

    if len(outs) != 1:
        problems.append(f"{len(outs)} outputs — expected 1 raw tensor "
                        "(multiple outputs usually means NMS was baked in)")
    else:
        shape = next(iter(outs.values()))
        anchors = shape[2] if len(shape) == 3 else "?"
        # yolo-generic wants [1, 4+nc, anchors]; reject [1, N, 6] NMS output.
        if len(shape) != 3 or shape[1] != 4 + nc:
            problems.append(
                f"output {shape} != [1, {4+nc}, anchors] (4 bbox + {nc} classes). "
                "If it looks like [1, N, 6], NMS is embedded — re-export with nms=False")

    if problems:
        print("[verify] DROP-IN CONTRACT MISMATCH:")
        for p in problems:
            print(f"[verify]   - {p}")
        raise SystemExit("[verify] Frigate yolo-generic would mis-parse this ONNX.")
    print(f"[verify] OK — drop-in compatible: "
          f"[1,3,{imgsz},{imgsz}] -> [1,{4+nc},{anchors}]")


if __name__ == "__main__":
    main()
