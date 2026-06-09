# Project: yolov9 (Ultralytics)

The first zoo candidate. **Lives in-place at the repo root** — the root pipeline
(`configs/`, `scripts/`) *is* the yolov9 project; it can move under here later.

## Where its pieces are
| Concern | Location |
|---|---|
| Config | `configs/train.yaml`, `configs/classes.yaml`, `configs/datasets.yaml` |
| Train | `scripts/train.py` (Ultralytics, checkpoint + auto-resume) → `make train` |
| Export | `scripts/export_onnx.py` → contract-verified ONNX |
| Benchmark | `scripts/eval_baseline.py` / `make baseline` (+ shared report) |
| Deploy | [`frigate/DEPLOY.md`](../../frigate/DEPLOY.md) runbook + `frigate/config.snippet.yaml` |
| Monitoring | self-hosted wandb (`scripts/ensure_wandb.py`, `scripts/wandb_logging.py`) |

## Frigate deploy contract
- `model_type: yolo-generic`, **640×640**, NCHW, FP32, RGB
- output `[1, 85, 8400]` (4 bbox + 81 classes), no embedded NMS
- labelmap: 81 lines, COCO-80 canonical order + `package` (id 80)

## Evaluate it yourself
This repo gives you the tools to measure — not a verdict. Train, then form your
own conclusions on your data:
- `make baseline` — COCO retention vs stock yolov9s (the no-forgetting check).
- `make benchmark` — package precision/recall on the held-out test split;
  `--scenario package_gold` runs it on your own reserved porch frames (the
  trustworthy, in-domain number). See `configs/benchmarks.yaml`.
