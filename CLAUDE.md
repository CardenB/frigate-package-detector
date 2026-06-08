# CLAUDE.md

## What this is

Fine-tunes **YOLOv9** to add a **`package`** class (absent from COCO) as a
**drop-in replacement for Frigate's default YOLOv9 ONNX**, **without
catastrophic forgetting**. The output keeps all **80 COCO classes in their exact
default order** and appends `package` as class 80 (output `[1, 85, anchors]`),
so Frigate's labelmap just gains one line and every existing class keeps working.

The non-trivial work is dataset assembly: several incompatible package datasets
plus COCO replay are remapped into one unified class space (COCO-80 + package),
merged, and split.

## The one principle that drives every decision: it's a drop-in replacement

The exported ONNX must present the **same I/O contract** as the model it
replaces, or Frigate mis-parses it. The deployment-specific values live in
`configs/` — treat those as the source of truth; do **not** hardcode a
particular deployment elsewhere.

- **Train at the input resolution the detector feeds the model** (`train.yaml:
  imgsz`). Training and deploy size must match — a mismatch quietly costs
  accuracy. Frigate `width/height` must equal `imgsz`.
- **Same model variant** as the stock model you're replacing (`train.yaml:
  model`, e.g. `yolov9s.pt`).
- **Class order is the contract.** Keep COCO ids 0..79 in canonical order and
  `package` last (`classes.yaml`). Reordering breaks Frigate's labelmap mapping.
- **Export interface**, enforced by `export_onnx.py` (keep this check):
  input `[1,3,imgsz,imgsz]` NCHW **FP32** RGB; output `[1, 4+nc, anchors]` with
  **no embedded NMS**. `export.half: false` keeps it FP32.

Everything else about a deployment — hardware/backend (CUDA, ROCm, Coral…),
container, cameras, zones — is plumbing that does **not** affect the finetune.
You should not need to know a deployment's specifics to use this repo; you only
need its model I/O contract, captured in `configs/`.

## Private local context

`.private/` (gitignored) holds machine-specific deployment notes that are NOT
shared. If it exists, **read it first** to build context on the actual target
(input size, variant, backend, quirks) — it's there to inform tuning.

Hard rule: its contents are **read-only context**. Never copy specifics from it
into tracked files, commit messages, PRs, or any committed config. Deployment
values that the finetune must match belong in `configs/` as ordinary defaults
(e.g. `imgsz`), expressed generically — not attributed to any deployment.

## Workflow

```
./bootstrap.sh            # venv + deps + (optional) build the dataset
make data                 # download -> convert -> build  => data/final/
make analyze              # class balance — run BEFORE training
make train                # fine-tune (configs/train.yaml)
make export               # -> ONNX + labelmap.txt, verifies the I/O contract
```

Stages are separable: `download_*` (slow, network) → `convert_to_yolo`
(normalize labels) → `build_dataset` (merge/split). Edit the class map and
re-run `make convert build` without re-downloading.

## Where to change things

- `configs/classes.yaml` — unified class list (index = YOLO id) + per-source
  label-name maps. Unmapped names are dropped. Edit, then `make convert build`.
- `configs/datasets.yaml` — which sources, sample caps, Roboflow projects.
- `configs/train.yaml` — variant, `imgsz`, hyperparams, export settings. **This
  file is where the deployment target is encoded** (size/variant) — keep it in
  sync with whatever you're replacing.

## Benchmarking (compare models on shared scenarios)

Two layers:
- `eval_baseline.py` (`make baseline`) — Ultralytics `val()` on COCO val; the
  catastrophic-forgetting reference for a YOLO model.
- `eval_remote.py` — detector-agnostic, **pycocotools**-scored benchmark that
  runs ANY detector (`--detector yolo|remote`) on a `--scenario` and writes a
  versioned run with TP/FP/FN galleries. This is what compares the finetune vs
  the remote open-vocab "locate-anything" model apples-to-apples (same data,
  vocab, scorer).

Scenarios are `(dataset, vocabulary)` presets in `configs/benchmarks.yaml`:
- `coco` — full COCO val, 80-class vocab (no package). Open-vocab vs YOLO.
- `package` — held-out `data/final` **test** split, security vocab incl.
  `package`. Same taxonomy as the finetune.

Run the matrix (skips models not yet available — safe at any stage):
```
python scripts/run_benchmarks.py --scenario package   # all models, one scenario
python scripts/run_benchmarks.py                      # all scenarios x models
python scripts/run_benchmarks.py --scenario coco --examples 0   # skip galleries
```
Runs land in `report/benchmarks/<schema>/<name>__<ts>/` (manifest + per_class +
gallery; schema/layout in `bench.py`); `make_benchmark_report.py` builds
`report/benchmark_report.{md,html}`.

Gotchas:
- **Remote = LocateAnything VLM** at `REMOTE_DETECTOR_URL` (.env), POST `/locate`
  (JSON: base64 image + `</c>`-joined prompt; pixel-xyxy `boxes` rescaled by the
  returned `image_size`; **no confidence**). ~1.2 s/img (int8) → use `--limit`
  for smoke; `--batch N` sends N imgs per `/locate_batch` forward (single-GPU, so
  batching not concurrency is the lever — but confirm it's actually faster with a
  single-vs-batch A/B before a full run). Full COCO is still long.
- **No confidence → mAP is degenerate** for the remote model (constant score).
  Compare **precision/recall/F1**, not mAP, for locate-anything.
- **`package` GT in `data/final` is the noisy training sources** (held-out
  images, not an independent domain). For a trustworthy package number use a
  reserved package set or the Frigate "gold" review rounds (see the `package`
  scenario caveat in `benchmarks.yaml`). COCO val has no package at all.

## Invariants & gotchas

- **Don't induce catastrophic forgetting.** Keep COCO replay (a sample spanning
  all 80 classes) in the training mix when adding package data — that's what
  preserves the existing classes. Don't trim it to a few classes.
- **`package` is the minority class.** Always `make analyze` first; if it's
  <~5% of boxes, add real package data (Roboflow) rather than leaning on the
  Open Images `Box` proxy, which is noisy (any cardboard/storage box).
- **Keep the export contract check.** A broken contract = Frigate mis-parses the
  model; fail loudly at export rather than discovering it on-device.
- **Optional accuracy lever:** if a target *stretches* non-square camera crops
  into the square model input (rather than letterboxing), training's default
  letterbox preprocessing mismatches it — consider matching the augmentation.
