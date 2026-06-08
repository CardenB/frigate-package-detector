# Review flywheel — turn Frigate detections into training data

Frames from your own cameras are the highest-value training data, and Frigate's
detections give you candidate boxes to *verify* rather than label from scratch.
Each round makes the model better, which makes the next round's candidates
better.

```
make frigate-pull  ─▶ pull recent detections + their boxes  ─▶ data/review/incoming/<round>/
make frigate-review─▶ verify/tag each frame in FiftyOne
make frigate-export─▶ gold YOLO round                       ─▶ data/raw/frigate/round-<id>/
make convert build ─▶ mixed in (whitelisted + oversampled x8)
make train         ─▶ better model ─▶ pull again next week
```

## One-time setup

Put your Frigate LAN URL in `.env` (private, never committed):

```
FRIGATE_URL=http://10.0.0.5:5000        # your Frigate
# FRIGATE_API_KEY=...                    # only if your Frigate needs auth
```

Test it: `make frigate-pull ARGS=--check` → should print the Frigate version.

## The loop

**1. Pull** — grab recent detections (defaults in `datasets.yaml: frigate.pull`):

```
make frigate-pull ARGS="--limit 100"
make frigate-pull ARGS="--labels package --after 2026-06-01 --cameras doorbell"
```

Pull low-score events too (`--min-score 0`, the default) — they're where the
false positives live, and confirmed FPs become hard negatives.

**2. Review** — opens FiftyOne in your browser with Frigate's boxes drawn:

```
make frigate-review                       # most recent round
make frigate-review ARGS="--round <id>"
```

Tag each frame with exactly one of:

| tag | meaning | becomes |
|---|---|---|
| `good` | box(es) are right | a positive training label |
| `negative` | false positive / nothing there | a **hard negative** (empty label) |
| `holdout` | verified-correct, reserve for eval | a **gold eval** frame in `data/gold_eval/` (never trained on) |
| `fix` | box wrong or object missed | skipped this round (edit later) |

Untagged frames are treated as not-reviewed and skipped. (Box *editing* via
CVAT can be added later; start verify-only — most of Frigate's boxes are fine.)

Reserve **~10–20%** of clean frames as `holdout` to grow a stable gold eval set
(`data/gold_eval/README.md`) — that's what makes your package metric trustworthy
round-over-round (`make benchmark ARGS="--scenario package_gold"`).

**3. Export** — writes the reviewed round to a gold dataset:

```
make frigate-export                       # most recent round
```

**4. Fold in & retrain:**

```
make convert build      # frigate is whitelisted + oversampled (see datasets.yaml)
make analyze            # confirm your frames are represented
make train && make export
```

## Notes

- **Oversampling**: `datasets.yaml: frigate.oversample` (default 8) duplicates your
  in-domain frames in *train only* so they aren't drowned by the big generic
  sets. Raise it while the round is small; lower it as it grows.
- **Hard negatives** are the strongest lever for cutting Frigate's false alarms —
  reject generously.
- **First pull is a sanity check**: do a small `--limit 20` and eyeball the boxes
  in review to confirm Frigate's box coordinates map correctly on your version
  (the human review step catches any mismatch before it reaches training).
- **Rounds are cumulative**: each export is a new `round-<id>/` under
  `data/raw/frigate/`; old rounds keep contributing. Delete a round dir to drop it.
