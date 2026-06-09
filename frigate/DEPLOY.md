# Deploy runbook — drop the `package` model into Frigate

How to take the model this repo produces and run it in Frigate as a **drop-in
replacement** for the default YOLOv9 ONNX: it adds a `package` class while every
existing COCO class keeps working. Includes a tested **rollback** path.

> Machine-specific values (host paths, your Frigate config location, the restart
> command) are yours to fill in. Keep them in a private, untracked note — this
> repo gitignores `.private/` and `deploy/` for exactly that. Don't commit them.

## What you deploy

`make export` writes these next to the trained weights (`models/<run>/weights/`)
and verifies the I/O contract:

| Asset | What it is |
|---|---|
| `best.onnx` | the model — `yolo-generic`, input `[1,3,640,640]` → output `[1,85,8400]` (NCHW, FP32, RGB) |
| `labelmap.txt` | 81 labels: COCO-80 in canonical order + `package` (id 80, last line) |

The Frigate config to merge is [`config.snippet.yaml`](config.snippet.yaml).

## Before you start

- **A package-capable model must be live before the review flywheel can start** —
  the stock model records zero `package` events, so v1 goes live first.
- **Train and deploy at the same input size:** `configs/train.yaml: imgsz` must
  equal Frigate's model `width`/`height` (default **640**).

## Steps

1. **Back up first (rollback prep).** Copy Frigate's *current* model file +
   labelmap + `config.yml` to a timestamped backup dir. Don't proceed until the
   backup is confirmed present.
2. **Sanity-check the ONNX** (recommended): load it with onnxruntime, feed a
   `[1,3,640,640]` float tensor, confirm output `[1,85,8400]` and finite values.
   If it won't load, **stop** — don't touch the live model.
3. **Copy assets** into Frigate's model dir (e.g. `/config/model_cache/`):
   `best.onnx` and `labelmap.txt`.
4. **Clear the compiled-model cache** if your detector compiles models (TensorRT
   `.engine`, MIGraphX `.mxr`, …) — otherwise Frigate reuses the *stale* compiled
   model instead of your new ONNX.
5. **Merge [`config.snippet.yaml`](config.snippet.yaml):** point `model.path` /
   `model.labelmap_path` at the new files, keep `model_type: yolo-generic`,
   `width`/`height` = your `imgsz`, NCHW/FP32/RGB, and add `package` to
   `objects.track`. Leave your existing `detectors:` block unchanged.
6. **Restart Frigate.** First boot may recompile the model (a few minutes).

## Verify

- Frigate starts cleanly and the model loads (check the logs).
- All 80 COCO classes still detect as before — **only `package` is new**.
- Await/stage a package on a relevant camera and confirm a `package` event. If
  none appears after a real delivery, re-check: assets copied, `package` in
  tracked objects, compiled cache cleared, and `width`/`height` match `imgsz`.

## Rollback

Point `model.path` / `model.labelmap_path` back at the backed-up previous model +
labelmap, clear the compiled cache again, and restart. (That's why step 1 is
non-negotiable.)

## Next: close the loop

Once it's live, package detections feed the **review flywheel**
([`FLYWHEEL.md`](FLYWHEEL.md)) — your own reviewed frames become gold training
data plus a held-out eval set, and each round improves the model.
