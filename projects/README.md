# Model zoo

Candidate detectors for the **same mission**: a drop-in package detector for
Frigate that keeps the existing COCO classes (no catastrophic forgetting). Each
candidate is **benchmarked → fine-tuned → deployed** through shared
infrastructure, so they're directly comparable.

| Project | Framework | Frigate `model_type` | Input | Fine-tune data | Status |
|---|---|---|---|---|---|
| [yolov9](yolov9/PROJECT.md) | Ultralytics | `yolo-generic` | 640×640 NCHW fp32 | YOLO (`data/final`) | **trained + deployed (v1)** — package mAP50 0.80 |
| [rfdetr](rfdetr/PROJECT.md) | `rfdetr` (Roboflow) | `rfdetr` | 320×320 NCHW fp32 | COCO-JSON (converted) | **candidate (scaffolded)** |

## What's shared vs. per-project

**Shared (repo root):**
- **Dataset** — `data/final/` (unified YOLO, 81 classes = COCO-80 + `package`),
  built once by `make data`. Class scheme in `configs/classes.yaml`.
- **Benchmark** — `report/benchmarks/<schema>/<run>/` + `make_benchmark_report.py`.
  Every model evaluates on the same scenarios (`configs/benchmarks.yaml`) so the
  side-by-side report compares them apples-to-apples.
- **The mission/contract** — drop-in for Frigate's default detector, all 80 COCO
  classes retained, `package` added.

**Per-project (`projects/<name>/`):** model-specific training, export, deploy
contract, and any data-format conversion (e.g. RF-DETR needs COCO-JSON, not YOLO).

## Notes
- **yolov9 lives in-place** (`configs/`, `scripts/`, `deploy/`) for now because a
  run is in flight; `projects/yolov9/PROJECT.md` points at it. It can migrate
  under `projects/yolov9/` later. RF-DETR starts clean in `projects/rfdetr/`.
- **Separate environments**: RF-DETR's deps (`rfdetr`/transformer DETR) differ from
  Ultralytics — each project gets its own venv (`projects/<name>/.venv`).
- **Comparing them**: register a model in `configs/benchmarks.yaml`, run the
  benchmark matrix, and read `report/benchmark_report.html`. The winner (accuracy
  vs. latency on the *actual* deploy hardware) is the one to ship.

## Add a new candidate
1. `mkdir projects/<name>/`, add a `PROJECT.md` (spec), `config.yaml`, env setup,
   and `train.py` / `export_onnx.py` for that framework.
2. If its labels aren't YOLO, add a converter from `data/final/`.
3. Register it in `configs/benchmarks.yaml` `models:` and add its detector path to
   the benchmark runner.
4. Define its Frigate deploy contract (model_type, input size, labelmap).
