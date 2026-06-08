# data/gold_eval/ — held-out gold evaluation set

**This directory is intentionally empty in the repo.** It fills up with *your own*
porch frames as you run the review flywheel — it is **never committed** (private
camera imagery) and **never trained on**.

## What it is

A stable, accumulating set of frames from your own cameras that you reserved for
**evaluation only**, so you can measure real-world package performance honestly
and **compare it round-over-round**. Because it shares no images with training,
its precision/recall actually reflect how the model does on your porch.

## How it gets populated

During a review round (`make frigate-review`), tag a clean frame **`holdout`**
instead of `good`. On `make frigate-export` those frames are written here:

```
data/gold_eval/
    images/<round>__<event>.jpg
    labels/<round>__<event>.txt     # YOLO; empty file = verified hard negative
    dataset.yaml                    # unified COCO-80 + package names
```

Reserve roughly **10–20%** of clean frames as `holdout`. Both positives and
verified negatives are valuable (negatives measure false positives).

## How it's used

```
make benchmark ARGS="--scenario package_gold"
```

`package_gold` (in `configs/benchmarks.yaml`) points here. It's **skipped
automatically until this set has data**, so it ships ready-but-dormant.

See `frigate/FLYWHEEL.md` for the full loop.
