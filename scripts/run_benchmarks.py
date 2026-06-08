#!/usr/bin/env python3
"""Run the model x scenario benchmark matrix from configs/benchmarks.yaml.

Each combo is scored identically (pycocotools) by eval_remote.py and written as a
versioned run under report/benchmarks/<schema>/; then make_benchmark_report.py
builds the side-by-side report. Combos that can't run yet are SKIPPED (not
errors), so this is safe to run at any stage:
  - a yolo model whose weights don't exist yet (finetune not trained)
  - the remote detector when REMOTE_DETECTOR_URL is unset

    python scripts/run_benchmarks.py                     # all scenarios x models
    python scripts/run_benchmarks.py --scenario package  # one scenario
    python scripts/run_benchmarks.py --limit 300         # cap images (slow VLM)
    python scripts/run_benchmarks.py --scenario coco --examples 0   # skip galleries
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

from common import CONFIGS, REPO

EVAL = REPO / "scripts" / "eval_remote.py"
REPORT = REPO / "scripts" / "make_benchmark_report.py"


def main() -> None:
    cfg = yaml.safe_load((CONFIGS / "benchmarks.yaml").read_text())
    scenarios, models = cfg.get("scenarios", {}), cfg.get("models", [])

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="one scenario (default: all)")
    ap.add_argument("--models", default=None,
                    help="comma list of model names to include (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="cap images per run")
    ap.add_argument("--examples", type=int, default=12, help="gallery top-K (0 to skip)")
    ap.add_argument("--no-report", action="store_true", help="skip report rebuild")
    args = ap.parse_args()

    sel_sc = [args.scenario] if args.scenario else list(scenarios)
    for sc in sel_sc:
        if sc not in scenarios:
            sys.exit(f"unknown scenario '{sc}' (have: {', '.join(scenarios)})")
    want_models = ({m.strip() for m in args.models.split(",")}
                   if args.models else None)
    py = sys.executable
    ran = 0

    for sc in sel_sc:
        # Skip a scenario whose dataset isn't present yet (e.g. package_gold
        # before you've reserved any `holdout` frames) — ready but dormant.
        sdata = Path(scenarios[sc]["data"])
        sdata = sdata if sdata.is_absolute() else REPO / sdata
        if not sdata.exists():
            print(f"[suite] SKIP scenario '{sc}': data not found ({scenarios[sc]['data']}) — populate it first")
            continue
        for m in models:
            if want_models and m["name"] not in want_models:
                continue
            name = f"{sc}__{m['name']}"
            cmd = [py, str(EVAL), "--scenario", sc, "--detector", m["detector"],
                   "--name", name, "--examples", str(args.examples)]

            if m["detector"] == "yolo":
                mp = Path(m["model"])
                mp = mp if mp.is_absolute() else REPO / mp
                # repo-relative weights (e.g. the finetune) may not exist yet;
                # bare names like yolov9s.pt auto-download, so don't skip those.
                if ("/" in m["model"]) and not mp.exists():
                    print(f"[suite] SKIP {name}: weights missing ({m['model']}) — train first")
                    continue
                cmd += ["--model", m["model"]]
            elif m["detector"] == "remote":
                if not os.environ.get("REMOTE_DETECTOR_URL"):
                    print(f"[suite] SKIP {name}: REMOTE_DETECTOR_URL unset (.env)")
                    continue

            if args.limit:
                cmd += ["--limit", str(args.limit)]
            print(f"\n[suite] >>> {name}")
            if subprocess.run(cmd).returncode == 0:
                ran += 1
            else:
                print(f"[suite] FAILED {name}")

    print(f"\n[suite] {ran} run(s) completed")
    if ran and not args.no_report:
        subprocess.run([py, str(REPORT)])


if __name__ == "__main__":
    main()
