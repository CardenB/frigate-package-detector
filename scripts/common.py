"""Shared helpers: config loading, paths, and the unified label remapper."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
DATA = REPO / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
FINAL = DATA / "final"
# Held-out gold eval set (your own reviewed frames, tagged `holdout`). Lives
# OUTSIDE data/raw so convert/build never train on it — used only by benchmarks
# to measure real-world porch performance round-over-round.
GOLD_EVAL = DATA / "gold_eval"


def _load_dotenv() -> None:
    """Load REPO/.env into os.environ (does not override already-set vars).

    Keeps every script working whether launched via `make` or directly, without
    a python-dotenv dependency.
    """
    env_path = REPO / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


_load_dotenv()


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def classes_cfg() -> dict:
    return load_yaml(CONFIGS / "classes.yaml")


def datasets_cfg() -> dict:
    return load_yaml(CONFIGS / "datasets.yaml")


def train_cfg() -> dict:
    return load_yaml(CONFIGS / "train.yaml")


def _curation_match(name: str, registry: dict):
    """Match a source name or roboflow dir against a whitelist/blacklist key.

    Keys may be a source ("package_seg") or a roboflow "workspace/project". A
    downloaded roboflow dir is "<ws>_<project>_v<N>", so we match the key with
    "/" normalized to "_" as a prefix.
    """
    for k, v in (registry or {}).items():
        kn = k.replace("/", "_")
        if name == k or name == kn or name.startswith(kn + "_"):
            return k, v
    return None


def curation(name: str) -> tuple[str, str]:
    """('deny' | 'defer' | 'allow' | 'pending', reason) for a source or 'ws/project'.

    Driven by the whitelist/blacklist/deferred registry in datasets.yaml. 'deny'
    (blacklist) and 'defer' (deferred — left out for now, revisit later) both mean
    the pipeline must skip it; 'pending' means reviewed-status unknown (still
    used, but callers should warn).
    """
    cfg = datasets_cfg()
    for status, key in (("deny", "blacklist"), ("defer", "deferred"), ("allow", "whitelist")):
        m = _curation_match(name, cfg.get(key) or {})
        if m:
            v = m[1]
            reason = (v.get("reason", "") if isinstance(v, dict) else str(v or "")).strip()
            return status, reason
    return "pending", ""


def build_mode() -> str:
    """'multi' (COCO classes + package) or 'single' (package only)."""
    return os.environ.get("BUILD_MODE", "multi").lower()


def unified_classes() -> list[str]:
    cfg = classes_cfg()
    if build_mode() == "single":
        return list(cfg["single_class_keep"])
    return list(cfg["classes"])


def class_index() -> dict[str, int]:
    return {name: i for i, name in enumerate(unified_classes())}


class Remapper:
    """Maps a single source's native class names -> unified class ids.

    Returns None for native classes that should be dropped (not in the mapping,
    or not part of the active build mode's class set).
    """

    def __init__(self, source: str):
        cfg = classes_cfg()
        if source not in cfg["sources"]:
            raise KeyError(f"No mapping for source '{source}' in classes.yaml")
        raw = dict(cfg["sources"][source])
        # `__identity__: true` -> native names that already equal a unified
        # class map to themselves (used for COCO, whose names match 0..79).
        self._identity = bool(raw.pop("__identity__", False))
        self._map = {k.lower(): v for k, v in raw.items()}
        self._idx = class_index()  # respects build mode
        self._canon = {c.lower(): c for c in unified_classes()}

    def to_id(self, native_name: str) -> int | None:
        key = native_name.strip().lower()
        unified = self._map.get(key)
        if unified is None and self._identity:
            unified = self._canon.get(key)  # native name IS a unified class
        if unified is None:
            return None
        return self._idx.get(unified)  # None if unified class not in this build
