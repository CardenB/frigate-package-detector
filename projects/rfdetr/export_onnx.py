#!/usr/bin/env python3
"""Export a fine-tuned RF-DETR checkpoint to ONNX for Frigate (`model_type: rfdetr`).

Run in projects/rfdetr/.venv.  rfdetr's .export() signature is known:
  export(output_dir, shape=(W,H), opset_version=17, format='onnx', simplify=...,
         quantization=..., ...). We set shape from config.resolution.

    python export_onnx.py [path/to/checkpoint]

[VERIFY] before deploying: load the resulting ONNX in onnxruntime and confirm the
I/O matches Frigate's rfdetr contract (input NCHW float at the export shape; the
DETR output the rfdetr handler expects). Confirm the labelmap order/indexing too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def main() -> None:
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    res = int(cfg["resolution"])
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    out_dir = HERE / "deploy"
    out_dir.mkdir(exist_ok=True)

    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
    Model = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[cfg["variant"]]
    model = Model(pretrain_weights=str(ckpt)) if ckpt else Model()  # [VERIFY] kwarg name

    print(f"[rfdetr] exporting ONNX shape=({res},{res}) -> {out_dir}")
    model.export(output_dir=str(out_dir), shape=(res, res),
                 opset_version=17, format=cfg["export"]["format"])

    # Labelmap = the lean class order the dataset was built with.
    classes_txt = HERE / cfg["coco_data_dir"] / "classes.txt"
    if classes_txt.exists():
        (out_dir / "labelmap.txt").write_text(classes_txt.read_text())
        print(f"[rfdetr] labelmap -> {out_dir/'labelmap.txt'}")
    print(f"[rfdetr] NEXT: verify ONNX I/O vs Frigate's rfdetr contract "
          f"(model_type: rfdetr, width/height = {res}, NCHW float). [VERIFY] labelmap indexing.")


if __name__ == "__main__":
    main()
