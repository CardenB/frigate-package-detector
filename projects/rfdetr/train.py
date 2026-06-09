#!/usr/bin/env python3
"""Fine-tune RF-DETR (lean security+package set) — run in projects/rfdetr/.venv.

    python convert_to_coco.py          # build the lean COCO-JSON first
    python train.py                    # full recipe (config.yaml)
    python train.py --epochs 1         # smoke test

rfdetr's train()/__init__ take **kwargs (params pass through to internal config),
so we assemble the documented recipe kwargs from config.yaml. num_classes is
auto-detected from the 12-category COCO json. [VERIFY] after a smoke run that the
EMA/early-stop kwargs are honored and the saved checkpoint records 12 classes
(known rfdetr export bug, issue #407).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, help="override epochs (smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    data_dir = (HERE / cfg["coco_data_dir"]).resolve()
    out_dir = (HERE / cfg["output_dir"]).resolve()
    if not (data_dir / "train" / "_annotations.coco.json").exists():
        raise SystemExit(f"No COCO data at {data_dir}. Run convert_to_coco.py first.")
    classes = (data_dir / "classes.txt").read_text().split()
    print(f"[rfdetr] {len(classes)} classes: {classes}")

    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
    Model = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[cfg["variant"]]
    model = Model()  # [VERIFY] does resolution belong here vs. in train()?

    train_kwargs = dict(
        dataset_dir=str(data_dir),
        epochs=args.epochs or cfg["epochs"],
        batch_size=cfg["batch_size"],
        grad_accum_steps=cfg["grad_accum_steps"],
        lr=cfg["lr"],
        resolution=cfg["resolution"],            # [VERIFY] arg name / placement
        output_dir=str(out_dir),
        early_stopping=cfg.get("early_stopping", True),
        early_stopping_patience=cfg.get("early_stopping_patience", 15),
        early_stopping_min_delta=cfg.get("early_stopping_min_delta", 0.005),
        early_stopping_use_ema=cfg.get("early_stopping_use_ema", True),
    )
    print(f"[rfdetr] variant={cfg['variant']} res={cfg['resolution']} "
          f"epochs={train_kwargs['epochs']} "
          f"eff_batch={cfg['batch_size']*cfg['grad_accum_steps']}")
    model.train(**train_kwargs)
    print(f"[rfdetr] done -> {out_dir}  (export with export_onnx.py)")


if __name__ == "__main__":
    main()
