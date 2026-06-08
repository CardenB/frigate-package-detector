#!/usr/bin/env python3
"""Fine-tune YOLOv9s on the merged dataset (configs/train.yaml), with robust
checkpointing + auto-resume.

Starts from COCO-pretrained yolov9s.pt: backbone transfers, detect head is
reinitialized for our 81-class set; the frozen backbone + COCO replay protect
the existing classes.

Robustness (important for long runs):
- Ultralytics writes `last.pt` every epoch; `save_period` (train.yaml) also keeps
  milestone snapshots. So a crash loses at most the in-progress epoch.
- This driver AUTO-RESUMES from `last.pt` on a crash (e.g. a transient DataLoader
  read failure under disk contention), up to --max-retries times.
- `--resume` continues a run after a full process death / reboot.

Examples:
    python scripts/train.py                       # full run per train.yaml
    python scripts/train.py --epochs 1            # quick smoke
    python scripts/train.py --resume              # continue the named run
    python scripts/train.py --max-retries 5       # more crash retries
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import wandb_logging
from common import FINAL, REPO, classes_cfg, train_cfg
from ensure_wandb import ensure_wandb
from ultralytics import YOLO

RETRY_BACKOFF_S = 10


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune per configs/train.yaml; "
                                 "flags override config and control resume.")
    ap.add_argument("--epochs", type=int, help="override epochs (e.g. 1 smoke test)")
    ap.add_argument("--batch", type=int, help="override batch size")
    ap.add_argument("--fraction", type=float, help="train on a fraction of the data")
    ap.add_argument("--name", help="override run name")
    ap.add_argument("--device", help="override device (e.g. 0, cpu)")
    ap.add_argument("--resume", action="store_true",
                    help="resume the named run from its last.pt")
    ap.add_argument("--max-retries", type=int, default=3,
                    help="auto-resume-from-checkpoint attempts on crash (0 = off)")
    ap.add_argument("--log-every", type=int, default=50,
                    help="wandb: log train loss/lr every N batches")
    ap.add_argument("--img-every", type=int, default=5,
                    help="wandb: log train sample images every N epochs")
    args = ap.parse_args()

    cfg = train_cfg()
    data_yaml = FINAL / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit("data/final/data.yaml missing. Run build_dataset.py first.")

    # Deterministic run dir so we can always find last.pt for resume.
    name = args.name or cfg.get("name", "yolov9s-package")
    project = cfg.get("project", "models")
    if not os.path.isabs(project):
        project = str(REPO / project)
    run_dir = Path(project) / name
    last_pt = run_dir / "weights" / "last.pt"

    # Start our own wandb run (dense scalars + val/train image examples). This
    # disables Ultralytics' built-in per-epoch wandb logging to avoid step
    # conflicts. wandb project name mirrors Ultralytics' default derivation.
    wb_run = None
    if ensure_wandb():
        wb_run = wandb_logging.start_run(name, project.replace("/", "-"))
    security = classes_cfg().get("security_classes", [])
    val_imgs_dir = str(FINAL / "val" / "images")

    # Fresh-run kwargs (resume ignores these — it reuses the saved args).
    kwargs = {k: v for k, v in cfg.items() if k not in ("model", "export")}
    kwargs.update(data=str(data_yaml), project=project, name=name, exist_ok=True)
    for key in ("epochs", "batch", "fraction", "device"):
        val = getattr(args, key)
        if val is not None:
            kwargs[key] = val

    print(f"[train] model={cfg['model']} data={data_yaml} run={run_dir}")
    print(f"[train] imgsz={kwargs.get('imgsz')} epochs={kwargs.get('epochs')} "
          f"batch={kwargs.get('batch')} freeze={kwargs.get('freeze')} "
          f"save_period={kwargs.get('save_period')} max_retries={args.max_retries}")

    attempt = 0
    while True:
        resume = (args.resume or attempt > 0) and last_pt.exists()
        try:
            if resume:
                print(f"[train] resuming from {last_pt}")
                model = YOLO(str(last_pt))
                wandb_logging.attach_callbacks(model, wb_run, args.log_every,
                                               args.img_every, per_class=security,
                                               val_dir=val_imgs_dir)
                results = model.train(resume=True)
            else:
                if args.resume:
                    print(f"[train] --resume set but no checkpoint at {last_pt}; "
                          "starting fresh")
                model = YOLO(cfg["model"])
                wandb_logging.attach_callbacks(model, wb_run, args.log_every,
                                               args.img_every, per_class=security,
                                               val_dir=val_imgs_dir)
                results = model.train(**kwargs)
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 — crash recovery is the point
            attempt += 1
            if attempt > args.max_retries:
                print(f"[train] FAILED after {attempt - 1} retries: {e}")
                raise
            how = "resume from checkpoint" if last_pt.exists() else "restart fresh"
            print(f"[train] crashed: {e}")
            print(f"[train] auto-recovery ({how}), retry {attempt}/{args.max_retries} "
                  f"in {RETRY_BACKOFF_S}s ...")
            time.sleep(RETRY_BACKOFF_S)

    print(f"[train] done. best weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
