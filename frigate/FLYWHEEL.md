# Iterative fine-tuning via the review flywheel

This guide covers the loop **you** run to keep improving the model with data from
your own cameras. The *training step itself* (`make train`) is a separate
handoff — see "Handoff with the training step" at the bottom. Your job is
everything around it: deploy, collect, review, fold in, repeat.

```
 (training step) ──▶ best.onnx + labelmap.txt
        │
   1. DEPLOY to Frigate ──▶ Frigate can now detect `package`
        │
   2. CONFIRM detections are flowing
        │
   3. REVIEW ROUND   make frigate-pull → frigate-review → frigate-export
        │            (verify boxes, reject false positives → hard negatives)
        │
   4. FOLD IN        make convert build   (frigate source: whitelisted, oversampled)
        │
        └──▶ hand back to the training step for vN+1 ──▶ (loop to 1)
```

Each turn makes the model better, which makes the next round's candidate boxes
better — less to correct, more value per minute of review.

---

## Prerequisite: a deployed model (the cold-start)

**The flywheel cannot start until a package-capable model is deployed to
Frigate.** The stock Frigate model has no `package` class, so Frigate records
*zero* package events and `frigate-pull` returns nothing. So **v1 must go live
first** (trained on the curated generic datasets), and only then does the loop
have real detections to review.

You need two files from the training step:
- `models/<run>/weights/best.onnx`
- `models/<run>/weights/labelmap.txt`  (81 lines: COCO-80 + `package`)

`make export` produces both and verifies the drop-in I/O contract.

---

## Step 1 — Deploy the model to Frigate

Follow the [deploy runbook](DEPLOY.md) (full steps + rollback). In short:

1. Copy `best.onnx` + `labelmap.txt` into Frigate's model dir.
2. Point `model.path` / `model.labelmap_path` at them. Keep the existing
   detector block and input fields (size/layout) unchanged.
3. Add `package` to your tracked objects (and any camera-level `track` lists).
4. If your detector compiles models (TensorRT `.engine`, MIGraphX `.mxr`, …),
   **clear that compiled cache** or the old model is reused.
5. Restart Frigate (first boot may recompile the model — a few minutes).

> Resolution must match what you trained at (`configs/train.yaml: imgsz` ==
> Frigate `width/height`).

---

## Step 2 — Confirm packages are being detected

```
make frigate-pull ARGS="--check"        # connectivity
```
Then watch Frigate's UI for `package` events on the porch/delivery cameras, or
do a tiny pull (Step 3) and confirm frames come back. If nothing appears after a
real delivery, re-check Step 1 (tracked objects + model swap + cache clear).

---

## Step 3 — Run a review round

Full command reference is in `frigate/REVIEW.md`; the essentials:

```
make frigate-pull   ARGS="--limit 50"        # small first; grow later
make frigate-review                          # opens FiftyOne; tag each frame
make frigate-export                          # → data/raw/frigate/round-<id>/
```

Tagging (one per frame):
- **good** — box(es) right → becomes a positive training label
- **negative** — false positive / nothing there → becomes a **hard negative**
- **holdout** — verified-correct, but **reserved for evaluation** → goes to
  `data/gold_eval/` and is **never trained on** (see "Quality gates" below)
- **fix** — wrong/missed box → skipped this round (defer to box editing)

Three habits that pay off:
- **Pull low-score events** (`--min-score 0`, the default). Those are where the
  false positives live, and confirmed FPs are the strongest signal for cutting
  Frigate's false alarms.
- **Reject generously.** Hard negatives are as valuable as positives here.
- **Reserve ~10–20% as `holdout`** every round. This builds a stable gold eval
  set so your package metric is honest and comparable over time.

---

## Step 4 — Fold in, then hand off for retraining

```
make convert build      # frigate is whitelisted + oversampled (datasets.yaml)
make analyze            # confirm your frames show up in the class counts
make balance            # see train balance before vs after oversampling
```

`data/final/` is now rebuilt with your reviewed round mixed in. **Hand that to
the training step** (next section). When it returns a new `best.onnx` +
`labelmap.txt`, go back to **Step 1** and redeploy.

---

## Cadence & tuning

- **Frequency:** a round whenever enough new package events have accumulated —
  often weekly early on. More frequent early (the model is weak); taper later.
- **Oversample:** `datasets.yaml: frigate.oversample` (default 8) duplicates your
  in-domain frames in *train only* so they aren't drowned by the generic sets.
  Keep it high while the round is small; lower it as your gold set grows past a
  few hundred frames.
- **Rounds are cumulative:** each export is a new `round-<id>/` under
  `data/raw/frigate/`; old rounds keep contributing. Delete a round dir to drop
  it. Re-export overwrites a round in place.

---

## Quality gates — what to watch each iteration

- **No forgetting:** `make baseline` evaluates the model's COCO classes. The
  80 COCO numbers must not drop vs the stock reference — that's the whole
  "without catastrophic forgetting" guarantee. If they slip, COCO replay is too
  small relative to package data (raise replay or lower frigate oversample).
- **Package quality:** `make benchmark ARGS="--scenario package_gold"` —
  precision/recall on **your own `holdout` frames** in `data/gold_eval/` (never
  trained on, accumulates across rounds), so the number is trustworthy and
  comparable round-over-round. (The plain `package` scenario reuses training
  sources, so treat it as a sanity check only.) `package_gold` is skipped until
  you've reserved some holdout frames.
- **False positives on your cameras:** the truest test is the next review
  round — fewer junk detections to reject means it's improving.

Stop iterating when a review round is mostly "good" with few FPs and the
benchmark P/R has plateaued.

---

## Handoff with the training step (a different agent owns this)

| | You (this guide) | Training step |
|---|---|---|
| **Produces** | rebuilt `data/final/` (incl. reviewed rounds), deploys ONNX, review rounds | `models/<run>/weights/best.onnx` + `labelmap.txt` |
| **Consumes** | `best.onnx` + `labelmap.txt` | `data/final/data.yaml` |
| **Commands** | `make convert build`, `make frigate-*`, deploy | `make train` → `make export` |

Trigger the training step after **Step 4** (dataset rebuilt). Tell them whether
to **fine-tune fresh from `yolov9s.pt`** (cleaner, avoids compounding drift —
preferred for the drop-in guarantee) or **resume** from the previous best
(faster, riskier for forgetting). When they hand back the new weights, you
redeploy (Step 1) and the loop continues.
