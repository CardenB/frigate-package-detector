#!/usr/bin/env python3
"""Download Roboflow Universe projects into data/raw/roboflow/<ws>_<proj>_v<n>/.

Needs ROBOFLOW_API_KEY (free account) in the environment / .env. Each project is
downloaded in YOLO format (train/valid/test + data.yaml). Add projects in
configs/datasets.yaml.
"""
from __future__ import annotations

import os
from pathlib import Path

from common import RAW, datasets_cfg


def main() -> None:
    cfg = datasets_cfg().get("roboflow", {})
    if not cfg.get("enabled", True):
        print("[roboflow] disabled in datasets.yaml — skipping")
        return

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("[roboflow] ROBOFLOW_API_KEY not set — skipping. "
              "Get a free key at https://app.roboflow.com/settings/api and put "
              "it in .env")
        return

    from roboflow import Roboflow

    fmt = cfg.get("format", "yolov9")
    projects = cfg.get("projects", [])
    rf = Roboflow(api_key=api_key)

    base = RAW / "roboflow"
    base.mkdir(parents=True, exist_ok=True)

    for p in projects:
        ws, proj, ver = p["workspace"], p["project"], int(p["version"])
        slug = f"{ws}_{proj}_v{ver}"
        location = base / slug
        if (location / "data.yaml").exists():
            print(f"[roboflow] {slug} already present — skipping")
            continue
        print(f"[roboflow] downloading {ws}/{proj} v{ver} ({fmt}) ...")
        try:
            version = rf.workspace(ws).project(proj).version(ver)
            version.download(fmt, location=str(location), overwrite=True)
            print(f"[roboflow]   -> {location}")
        except Exception as e:  # noqa: BLE001 — keep going on a bad project
            print(f"[roboflow]   FAILED {slug}: {e}")

    print(f"[roboflow] done -> {base}")


if __name__ == "__main__":
    main()
