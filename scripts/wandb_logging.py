"""Higher-frequency wandb logging for Ultralytics training.

Ultralytics' built-in wandb integration logs once per epoch with an explicit
`step=epoch+1`, only the 81-class AGGREGATE metrics, and saves val prediction
images only at the very end. That's too coarse and hides per-class (package)
progress. So we TAKE OVER and log on ONE monotonic global step:

  - train loss + lr every N batches            (dense scalars)
  - aggregate val metrics every epoch          (mAP / P / R)
  - PER-CLASS val metrics for security classes every epoch  (e.g. val/package/*)
  - val PREDICTION examples every epoch        (generated here, incl. package frames)
  - train sample images every K epochs         (augmented mosaics)
  - final summary plots at end

Step = epoch*MULT + batch, so it's correct across auto-resume. Callbacks are
DEFENSIVE — any logging error is swallowed so it can never crash training.
"""
from __future__ import annotations

from pathlib import Path

_STEP_MULT = 1_000_000


def start_run(run_name: str, project: str):
    """Init our own wandb run and stop Ultralytics from logging. Returns run or None."""
    try:
        import wandb
        from ultralytics.utils import SETTINGS
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] logging setup skipped: {e}")
        return None
    try:
        SETTINGS.update({"wandb": False})  # we own logging now
    except Exception:
        pass
    try:
        return wandb.init(project=project, name=run_name, id=run_name, resume="allow")
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] init failed, continuing without logging: {e}")
        return None


def _pick_val_images(val_dir: Path, n: int) -> list[str]:
    """A fixed, deterministic sample of val images — bias toward ones containing
    `package` GT (class id 80) so package predictions are visible, plus a spread
    of the rest. Same images every epoch => you watch the model improve on them."""
    imgs = sorted(val_dir.glob("*.jpg")) if val_dir.exists() else []
    if not imgs:
        return []
    lbl_dir = Path(str(val_dir).replace("/images", "/labels"))
    pkg, other = [], []
    for p in imgs:
        if len(pkg) >= n // 2:
            break
        lf = lbl_dir / f"{p.stem}.txt"
        try:
            if lf.exists() and any(ln.split()[:1] == ["80"]
                                   for ln in lf.read_text().splitlines() if ln.strip()):
                pkg.append(p)
        except Exception:
            pass
    # spread of the rest (evenly spaced across the sorted list)
    step = max(1, len(imgs) // max(1, n))
    other = [p for p in imgs[::step] if p not in pkg]
    picked = (pkg + other)[:n]
    return [str(p) for p in picked]


def attach_callbacks(model, run, log_every: int = 50, img_every_epochs: int = 5,
                     max_imgs: int = 6, per_class=None, val_dir=None) -> None:
    """Attach dense scalar + per-class + image-example logging callbacks."""
    if run is None:
        return
    import wandb

    keep = {c.lower() for c in (per_class or [])}
    val_imgs = _pick_val_images(Path(val_dir), max_imgs) if val_dir else []
    state = {"bi": 0, "warned": False}

    def gstep(trainer) -> int:
        return int(trainer.epoch) * _STEP_MULT + state["bi"]

    def safe(fn):
        def wrapped(trainer):
            try:
                fn(trainer)
            except Exception as e:  # noqa: BLE001 — never crash training
                if not state["warned"]:
                    print(f"[wandb] logging error (continuing without it): {e}")
                    state["warned"] = True
        return wrapped

    def on_epoch_start(trainer):
        state["bi"] = 0

    def on_batch_end(trainer):
        state["bi"] += 1
        if state["bi"] % log_every:
            return
        d = {k: float(v) for k, v in
             trainer.label_loss_items(trainer.tloss, prefix="train").items()}
        for k, v in (getattr(trainer, "lr", None) or {}).items():
            d[k] = float(v)
        wandb.log(d, step=gstep(trainer))

    def log_per_class(trainer, step):
        v = getattr(trainer, "validator", None)
        if v is None or not keep:
            return
        box = v.metrics.box
        names = getattr(v, "names", None) or trainer.data["names"]
        out = {}
        for idx, cid in enumerate(box.ap_class_index):
            nm = names[int(cid)]
            if nm.lower() in keep:
                p, r, ap50, ap = box.class_result(idx)
                out[f"val/{nm}/precision"] = float(p)
                out[f"val/{nm}/recall"] = float(r)
                out[f"val/{nm}/mAP50"] = float(ap50)
                out[f"val/{nm}/mAP50-95"] = float(ap)
        if out:
            wandb.log(out, step=step)

    def log_val_preds(trainer, step):
        if not val_imgs:
            return
        last = Path(trainer.save_dir) / "weights" / "last.pt"
        if not last.exists():
            return
        from ultralytics import YOLO
        res = YOLO(str(last)).predict(val_imgs, imgsz=int(trainer.args.imgsz),
                                      conf=0.25, device=0, verbose=False)
        for i, r in enumerate(res):
            # r.plot() returns BGR; wandb wants RGB
            wandb.log({f"val_pred/img{i}": wandb.Image(r.plot()[..., ::-1])}, step=step)

    def log_train_imgs(trainer, step):
        for p in sorted(Path(trainer.save_dir).glob("train_batch*.jpg"))[:max_imgs]:
            wandb.log({f"train_examples/{p.stem}": wandb.Image(str(p))}, step=step)

    def on_fit_epoch_end(trainer):
        step = gstep(trainer)
        wandb.log({k: float(v) for k, v in (trainer.metrics or {}).items()}, step=step)
        log_per_class(trainer, step)
        log_val_preds(trainer, step)                       # every epoch ("on each val")
        if int(trainer.epoch) % max(1, img_every_epochs) == 0:
            log_train_imgs(trainer, step)

    def on_train_end(trainer):
        step = gstep(trainer)
        try:
            for tag in ("PR_curve", "confusion_matrix", "results", "F1_curve"):
                for p in sorted(Path(trainer.save_dir).glob(f"*{tag}*.png"))[:2]:
                    wandb.log({f"summary/{p.stem}": wandb.Image(str(p))}, step=step)
        finally:
            wandb.finish()

    model.add_callback("on_train_epoch_start", on_epoch_start)
    model.add_callback("on_train_batch_end", safe(on_batch_end))
    model.add_callback("on_fit_epoch_end", safe(on_fit_epoch_end))
    model.add_callback("on_train_end", safe(on_train_end))
    print(f"[wandb] dense logging: scalars/{log_every} batches, per-class "
          f"{sorted(keep)}, val preds/epoch ({len(val_imgs)} imgs), "
          f"train imgs/{img_every_epochs} epochs")
