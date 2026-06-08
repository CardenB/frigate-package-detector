# frigate-package-detector

Fine-tune **YOLOv9** to add a **`package`** class (which COCO — and so Frigate's
default model — lacks), as a **drop-in replacement for Frigate's default YOLOv9
ONNX**, **without catastrophic forgetting** of the existing classes.

The output keeps all **80 COCO classes in their exact default order** and
appends `package` as class 80. So Frigate's existing labelmap just gains one
line and every other class keeps working — you only add `package` to your
tracked objects.

The hard part isn't training; it's assembling one clean dataset from several
incompatible sources. This repo does that: it pulls package data + COCO replay
data, **remaps every source's labels into one unified class space** (COCO-80 +
package), merges and splits them, and trains/exports a Frigate-ready ONNX.

## How it avoids catastrophic forgetting

A drop-in must still detect everything the default model does. Two things keep
the 80 COCO classes alive through fine-tuning:

1. **COCO replay** — a broad COCO-2017 sample spanning all 80 classes is mixed
   into the training data (`configs/datasets.yaml: coco_subset`), so those
   classes keep getting reinforced while `package` is learned.
2. **Backbone freeze** — the COCO-pretrained backbone is frozen early
   (`configs/train.yaml: freeze`) to protect transferred features.

The unified class list is the canonical COCO-80 order + `package`
(`configs/classes.yaml`). Don't reorder 0..79 — that order is the drop-in
contract with Frigate's labelmap.

## Data sources

| Source | What it gives | Needs |
|---|---|---|
| **Ultralytics package-seg** | real package boxes (segmentation→bbox) | nothing |
| **Roboflow Universe** | porch/delivery/parcel projects | free `ROBOFLOW_API_KEY` |
| **Open Images V7** | high-volume person/vehicle/animal + `Box` proxy | nothing (FiftyOne) |
| **COCO 2017 replay** | keeps all 80 COCO classes alive (anti-forgetting) | nothing (FiftyOne) |

Tune which sources, classes, and sample caps in `configs/datasets.yaml`. Add
more Roboflow package projects there — the more real porch imagery, the better.

## Quick start

Easiest — one interactive script that asks for your Roboflow key, builds the
venv, and (optionally) downloads + builds the dataset:

```bash
cd ~/gitrepos/frigate-package-detector
./bootstrap.sh                  # paste Roboflow key when prompted; follow prompts
```

Then, after it finishes:

```bash
source .venv/bin/activate
make train                      # fine-tune yolov9s on the 5090
make export                     # -> ONNX + labelmap.txt for Frigate
```

<details><summary>Manual / step-by-step (what bootstrap.sh automates)</summary>

```bash
./setup.sh                      # venv + torch(cu128 for the 5090) + deps
source .venv/bin/activate
cp .env.example .env            # add ROBOFLOW_API_KEY (optional but recommended)

make data                       # download -> convert -> build  (data/final/)
make analyze                    # check class balance BEFORE training
make train                      # fine-tune yolov9s on the 5090
make export                     # -> ONNX + labelmap.txt for Frigate
```
</details>

Then follow `frigate/config.snippet.yaml` to wire it into Frigate.

> Scripts auto-load `.env` (via `scripts/common.py`), so the Roboflow key and
> `BUILD_MODE` apply whether you run through `make` or call a script directly.

## Pipeline

```
download_*.py   data/raw/<source>/      native labels, per source
   │  (fetch — slow, network)
convert_to_yolo.py  data/interim/<source>/   unified label ids, polygons→bbox
   │  (normalize — fast, re-run after editing classes.yaml)
build_dataset.py    data/final/{train,val,test}/ + data.yaml   stratified split
   │
train.py  →  models/yolov9s-package*/weights/best.pt
export_onnx.py  →  best.onnx + labelmap.txt
```

Each stage is its own `make` target so you can re-run just one (e.g. tweak the
class map and `make convert build` without re-downloading).

## Layout

```
configs/   classes.yaml (unified classes + per-source maps), datasets.yaml, train.yaml
scripts/   download_*, convert_to_yolo, build_dataset, analyze_dataset, train, export_onnx
data/      raw/ interim/ final/   (gitignored)
models/    training runs + exported onnx (gitignored)
frigate/   config.snippet.yaml
```

## Notes & gotchas

- **Training is independent of the deploy target.** You train wherever you have
  a GPU and deploy the ONNX to your Frigate box; they can be different machines.
  The only thing training must mirror is the model I/O contract (size + class
  order), which lives in `configs/`.
- **Compiled-model cache**: if your Frigate detector compiles models (TensorRT
  `.engine`, MIGraphX `.mxr`, …), clear that cache when swapping the ONNX or the
  stale compiled model is reused.
- **Open Images `Box`** is a *noisy* package proxy (any cardboard/storage box).
  It boosts volume but can cause false positives; disable via
  `open_images.map_box: false` if precision suffers. Real porch data from
  Roboflow is higher quality — prioritize adding those projects.
- **Class balance**: `make analyze` warns if `package` is <5% of boxes. If so,
  add Roboflow projects or lower the COCO/OpenImages caps in `datasets.yaml`.
- **Input size**: Frigate's default yolov9 ONNX runs at **640×640**, so
  `train.yaml:imgsz` defaults to 640. Train and deploy at the same size —
  `imgsz` and Frigate's `width/height` must agree.
- **Frigate config fields** (`model_type`, `input_dtype`, …) have shifted across
  Frigate releases — verify the snippet against *your* version's
  `object_detectors` docs. `model_type: yolo-generic` is the yolov9-ONNX path.
- **FiftyOne downloads** can be large; set `FIFTYONE_DATASET_ZOO_DIR` in `.env`
  to a roomy disk if needed.

## License

[MIT](LICENSE) — free to use, modify, and redistribute with attribution.
