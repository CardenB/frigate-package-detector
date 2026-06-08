#!/usr/bin/env python3
"""Log an existing benchmark run to the self-hosted wandb (back-fill).

    python scripts/log_benchmark_wandb.py                     # latest run
    python scripts/log_benchmark_wandb.py package__locate-anything   # latest by name
    python scripts/log_benchmark_wandb.py report/benchmarks/v1/<run>  # explicit dir
    python scripts/log_benchmark_wandb.py --all               # every v1 run

Idempotent: re-running updates the same wandb run (id = run folder name).
"""
from __future__ import annotations

import sys
from pathlib import Path

import bench
from bench_wandb import log_run


def all_runs() -> list[Path]:
    return sorted(r for r in bench.schema_dir().glob("*")
                  if (r / "manifest.json").exists())


def resolve(arg: str | None) -> Path:
    if arg:
        p = Path(arg)
        if p.is_dir():
            return p
        matches = [r for r in all_runs() if r.name.startswith(arg)]
        if not matches:
            sys.exit(f"no run dir matching '{arg}' under {bench.schema_dir()}")
        return matches[-1]
    runs = all_runs()
    if not runs:
        sys.exit(f"no benchmark runs under {bench.schema_dir()}")
    return max(runs, key=lambda r: (r / "manifest.json").stat().st_mtime)


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--all":
        runs = all_runs()
        if not runs:
            sys.exit("no runs to log")
        ok = sum(log_run(r) for r in runs)
        print(f"[wandb] logged {ok}/{len(runs)} runs")
        return
    run_dir = resolve(args[0] if args else None)
    print(f"[wandb] logging {run_dir.name}")
    raise SystemExit(0 if log_run(run_dir) else 1)


if __name__ == "__main__":
    main()
