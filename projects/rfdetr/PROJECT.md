# Project: rfdetr (Roboflow RF-DETR)

Second zoo candidate. A real-time **DETR** (transformer) detector, SOTA on COCO,
explicitly designed for fine-tuning, Apache-2.0, and **natively supported by
Frigate** (`model_type: rfdetr`). Same mission as yolov9: add `package` without
forgetting the COCO classes — but a different framework, data format, and deploy
contract, so it lives in its own project with its own env.

## ⚠️ Decide this FIRST: will it run on the deploy hardware?
RF-DETR is **heavier than yolov9s**. Frigate's docs recommend running it on a
**discrete GPU** (they call out Arc). If your Frigate target is a modest
**integrated GPU / edge detector** rather than a discrete GPU, RF-DETR may be
too slow there. So the **first task is a latency probe on the actual target**,
not training:
- Export a stock RF-DETR-Nano ONNX, deploy to Frigate, measure inference ms.
- If it's far slower than yolov9s on the same target, RF-DETR is a non-starter on
  that hardware and you'd need a discrete GPU / different host before investing
  in tuning.

Pick the variant accordingly: **RF-DETR-Nano** (2.3 ms on a good GPU, 67.6 AP50)
for edge/real-time; Medium for accuracy if the hardware allows.

## Plan: benchmark → fine-tune → deploy
1. **Benchmark (stock)** — register `rfdetr-stock` in `configs/benchmarks.yaml`,
   eval on the shared `coco` + `package` scenarios. Confirms the harness + gives a
   baseline. (Needs an rfdetr detector path in the benchmark runner — see TODO.)
2. **Fine-tune** — convert `data/final` → COCO-JSON (`convert_to_coco.py`), train
   with the `rfdetr` package to add `package` (COCO-80 + package, COCO replay for
   no-forgetting), export ONNX.
3. **Deploy** — bundle ONNX + labelmap + a Frigate `rfdetr` config snippet, with
   the same backup/rollback flow as `deploy/`.

## Environment (separate venv — deps differ from Ultralytics)
```bash
projects/rfdetr/setup.sh          # creates projects/rfdetr/.venv + installs rfdetr
```

## Data: YOLO → COCO-JSON
RF-DETR wants `train/ valid/ test/` each with images + `_annotations.coco.json`.
```bash
projects/rfdetr/.venv/bin/python projects/rfdetr/convert_to_coco.py
# reads repo data/final/{train,val,test} (YOLO) + configs/classes.yaml
# writes projects/rfdetr/data/{train,valid,test}/_annotations.coco.json (+ image links)
```
Categories = COCO-80 + `package` (81), same scheme as yolov9, so retention works
the same way (the COCO replay already in `data/final` carries over).

## Train (sketch — see train.py)
```python
from rfdetr import RFDETRNano
RFDETRNano().train(dataset_dir=".../projects/rfdetr/data", epochs=..., 
                   batch_size=4, grad_accum_steps=4, lr=1e-4, resolution=...,
                   output_dir=".../projects/rfdetr/runs")
# total batch (batch_size * grad_accum_steps) should be ~16
```

## Frigate deploy contract
- `model_type: rfdetr`, **NCHW, float**, input **[VERIFY size]** (Frigate docs show
  320×320; RF-DETR training resolution must be **divisible by 56**, and 320 is NOT
  — so the export/Frigate `width`/`height` must equal whatever you actually export.
  Resolve this before deploying: pick a 56-divisible train/export size and set
  Frigate to match, or confirm Nano exports at 320.)
- labelmap: COCO-80 + `package`. **[VERIFY]** exact labelmap format/order Frigate's
  rfdetr post-processor expects (DETR class indexing can differ from YOLO; RF-DETR
  often reserves class 0 / uses 1-indexed COCO — confirm against a stock export).
- output tensor format is DETR-style (boxes + logits), parsed by Frigate's `rfdetr`
  handler — no NMS. **[VERIFY]** against a known-good stock rfdetr ONNX in Frigate.

## Open questions / blocks to fill
- **[VERIFY] resolution**: training divisible-by-56 vs Frigate's 320 — what size to
  export at, and does Frigate accept it?
- **[VERIFY] pretrained + class expansion**: does `RFDETRNano()` start from
  COCO-pretrained weights, and how does it handle going to 81 categories while
  keeping the 80 COCO ones? (Confirms the no-forgetting approach transfers.)
- **[VERIFY] labelmap indexing** for Frigate rfdetr (0- vs 1-indexed, background slot).
- **[TODO] benchmark runner**: add an `rfdetr` detector path to `eval_remote.py`
  (load the ONNX, run inference, emit boxes in the shared scorer's format) so
  `run_benchmarks.py` can include it.
- **[FILL] deploy hardware**: confirm the Frigate GPU can run RF-DETR at acceptable
  latency (see hardware warning above).

## Files here
- `config.yaml` — variant, resolution, epochs, batch, paths
- `requirements.txt` / `setup.sh` — isolated env
- `convert_to_coco.py` — `data/final` (YOLO) → COCO-JSON (implemented)
- `train.py` / `export_onnx.py` — scaffolds (need the rfdetr env + the [VERIFY]s)

Sources: [RF-DETR repo](https://github.com/roboflow/rf-detr) ·
[RF-DETR train docs](https://rfdetr.roboflow.com/learn/train/) ·
[Frigate object detectors](https://docs.frigate.video/configuration/object_detectors/)
