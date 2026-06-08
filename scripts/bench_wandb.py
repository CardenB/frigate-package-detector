"""Live wandb logging for benchmark runs (self-hosted server).

A run is logged in three phases so it's visible WHILE the benchmark runs:
  init_run(config)            -> run appears immediately as "running"
  log_progress(run, metrics)  -> stream images-done / failures / latency
  finalize_run(run, run_dir)  -> per-class table, latency hist/table, examples; close

Safe no-op if wandb isn't ready (mirrors ensure_wandb), so callers never guard it.
log_run(run_dir) = init+finalize in one shot (used by the back-fill tool).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import bench
from ensure_wandb import ensure_wandb

MAX_EX_PER_CELL = 6      # cap example images per (class, category) sent to wandb
MAX_LAT_ROWS = 2000      # cap per-image latency rows sent to wandb

# manifest/config fields surfaced as wandb run config (for the runs comparison table)
_CONFIG_KEYS = ("detector", "mode", "scenario", "dataset", "split", "imgsz",
                "scorer", "model", "model_params", "num_images", "num_instances")


def init_run(config: dict, verbose: bool = True):
    """Start a live wandb run. Returns the run, or None if wandb isn't ready."""
    if not ensure_wandb(verbose=verbose):
        return None
    import wandb
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "frigate-package-detector"),
        name=config.get("name"),
        id=config.get("run_id", config.get("name")),  # stable id -> idempotent
        resume="allow",
        job_type="benchmark",
        group=config.get("scenario", "adhoc"),
        config={k: config.get(k) for k in _CONFIG_KEYS},
        reinit=True,
    )


def log_progress(run, metrics: dict, step: int | None = None) -> None:
    """Stream a progress point to the live run (no-op if wandb off)."""
    if run is not None:
        run.log(metrics, step=step)


def _log_final_media(run, run_dir: Path, manifest: dict, per_class: list) -> None:
    import wandb

    for k, v in (manifest.get("overall") or {}).items():
        run.summary[f"overall/{k}"] = v

    table = wandb.Table(
        columns=["class", "support", "precision", "recall", "map50", "map50_95"])
    for c in per_class:
        table.add_data(c["name"], c.get("support"), c.get("precision"),
                       c.get("recall"), c.get("map50"), c.get("map50_95"))
        if c.get("map50_95") is not None:
            run.summary[f"map50_95/{c['name']}"] = c["map50_95"]
    run.log({"per_class": table})

    # Latency: summary stats + histogram + per-image table
    for field, stats in (manifest.get("latency") or {}).items():
        if isinstance(stats, dict):
            for stat, v in stats.items():
                run.summary[f"latency/{field}/{stat}"] = v
    lat_path = run_dir / "latency.json"
    if lat_path.exists():
        per_img = json.loads(lat_path.read_text())
        key = "server_ms" if any(r.get("server_ms") is not None for r in per_img) else "wall_ms"
        vals = [r[key] for r in per_img if r.get(key) is not None]
        if vals:
            run.log({f"latency_hist/{key}": wandb.Histogram(vals)})
        lt = wandb.Table(columns=["image", "wall_ms", "server_ms",
                                  "inference_ms", "queue_wait_ms"])
        for r in per_img[:MAX_LAT_ROWS]:
            lt.add_data(r.get("image"), r.get("wall_ms"), r.get("server_ms"),
                        r.get("inference_ms"), r.get("queue_wait_ms"))
        run.log({"latency_per_image": lt})

    # Example images (TP/FP/FN) as an Image table, if the gallery rendered any
    ex_table = wandb.Table(columns=["class", "category", "image"])
    n_imgs = 0
    for c in per_class:
        ex = c.get("examples")
        if not ex:
            continue
        for cat in bench.CATEGORIES:
            for rel in (ex.get(cat) or [])[:MAX_EX_PER_CELL]:
                p = run_dir / rel
                if p.exists():
                    ex_table.add_data(c["name"], cat, wandb.Image(str(p)))
                    n_imgs += 1
    if n_imgs:
        run.log({"examples": ex_table})
    return n_imgs


def finalize_run(run, run_dir: Path, status: str = "finished",
                 verbose: bool = True) -> bool:
    """Log final media from the run dir and close the run. status != 'finished'
    marks it aborted/failed (e.g. circuit-breaker)."""
    if run is None:
        return False
    run_dir = Path(run_dir)
    n_imgs = 0
    mp = run_dir / "manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text())
        pc = run_dir / "per_class.json"
        per_class = json.loads(pc.read_text()) if pc.exists() else []
        n_imgs = _log_final_media(run, run_dir, manifest, per_class)
    if status != "finished":
        run.summary["status"] = status
    run.finish(exit_code=0 if status == "finished" else 1)
    if verbose:
        print(f"[wandb] {status} run '{run.name}' "
              f"({n_imgs} example imgs) -> "
              f"{os.environ.get('WANDB_BASE_URL', 'http://localhost:8080')}")
    return True


def log_run(run_dir: Path, verbose: bool = True) -> bool:
    """One-shot init+finalize for an already-complete run dir (back-fill)."""
    run_dir = Path(run_dir)
    mp = run_dir / "manifest.json"
    if not mp.exists():
        if verbose:
            print(f"[wandb] no manifest.json in {run_dir} — skip")
        return False
    manifest = json.loads(mp.read_text())
    cfg = dict(manifest)
    cfg["run_id"] = run_dir.name
    run = init_run(cfg, verbose=verbose)
    return finalize_run(run, run_dir, verbose=verbose)
