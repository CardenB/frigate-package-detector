# Project: yolov9 (Ultralytics)

The first zoo candidate, and the one currently deployed. **Lives in-place at the
repo root** (a training run is in flight, so it hasn't been moved under here).

## Where its pieces are
| Concern | Location |
|---|---|
| Config | `configs/train.yaml`, `configs/classes.yaml`, `configs/datasets.yaml` |
| Train | `scripts/train.py` (Ultralytics, checkpoint + auto-resume) → `make train` |
| Export | `scripts/export_onnx.py` → contract-verified ONNX |
| Benchmark | `scripts/eval_baseline.py` / `make baseline` (+ shared report) |
| Deploy bundle | `deploy/` (onnx + labelmap + `config.snippet.yaml` + `DEPLOY.md`) |
| Monitoring | self-hosted wandb (`scripts/ensure_wandb.py`, `scripts/wandb_logging.py`) |

## Frigate deploy contract
- `model_type: yolo-generic`, **640×640**, NCHW, FP32, RGB
- output `[1, 85, 8400]` (4 bbox + 81 classes), no embedded NMS
- labelmap: 81 lines, COCO-80 canonical order + `package` (id 80)

## Status
- Trained on full-COCO replay + Open Images + curated Roboflow (133k train).
- **Package** mAP50 **0.80** / mAP50-95 0.64 (in-domain test); COCO retention
  **−0.011** mAP50-95 vs stock (minimal forgetting).
- Deploy v1 bundle built (mid-training snapshot). Re-export
  `models/yolov9s-package/weights/best.pt` after the full run for the final model.
