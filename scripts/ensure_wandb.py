#!/usr/bin/env python3
"""Preflight: ensure the self-hosted wandb server is up before training.

Idempotent and SAFE — if anything is missing (docker, server, login) it prints
guidance and returns False so training continues WITHOUT wandb rather than
failing. Called automatically by train.py; also runnable directly:

    python scripts/ensure_wandb.py        # start/check server, print next steps

Server = `wandb/local` container in Docker Desktop on http://localhost:8080
(override with WANDB_BASE_URL). Set WANDB_MODE=disabled to skip entirely.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request

WANDB_URL = os.environ.get("WANDB_BASE_URL", "http://localhost:8080").rstrip("/")
CONTAINER = "wandb-local"
IMAGE = "wandb/local"
PROJECT = os.environ.get("WANDB_PROJECT", "frigate-package-detector")


# Docker Desktop's credsStore needs docker-credential-desktop.exe on PATH, which
# isn't there in a plain WSL shell — so even pulling a PUBLIC image fails. Inject
# the Windows Docker Desktop bin dir (override with DOCKER_DESKTOP_BIN).
_DESKTOP_BIN = os.environ.get(
    "DOCKER_DESKTOP_BIN",
    "/mnt/c/Program Files/Docker/Docker/resources/bin")


def _docker_env() -> dict:
    env = os.environ.copy()
    if os.path.isdir(_DESKTOP_BIN) and _DESKTOP_BIN not in env.get("PATH", ""):
        env["PATH"] = env.get("PATH", "") + os.pathsep + _DESKTOP_BIN
    return env


def _run(args, **kw):
    kw.setdefault("env", _docker_env())
    return subprocess.run(args, capture_output=True, text=True, **kw)


def container_state() -> str:
    """'running' | 'exited' | 'missing'."""
    r = _run(["docker", "inspect", "-f", "{{.State.Status}}", CONTAINER])
    return r.stdout.strip() if r.returncode == 0 else "missing"


def start_server() -> None:
    state = container_state()
    if state == "running":
        return
    if state == "missing":
        print(f"[wandb] creating {CONTAINER} container ({IMAGE}) ...")
        _run(["docker", "run", "-d", "--name", CONTAINER, "--restart",
              "unless-stopped", "-p", "8080:8080", "-v", "wandb-local:/vol",
              IMAGE]).check_returncode()
    else:
        print(f"[wandb] starting stopped {CONTAINER} container ...")
        _run(["docker", "start", CONTAINER]).check_returncode()


def wait_healthy(timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for path in ("/healthz", "/"):
            try:
                with urllib.request.urlopen(WANDB_URL + path, timeout=3) as resp:
                    if resp.status < 500:
                        return True
            except Exception:
                pass
        time.sleep(3)
    return False


def is_logged_in() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    # wandb writes the netrc machine as host:port (e.g. "localhost:8080"), but
    # some flows use just the host — accept either.
    p = urllib.parse.urlparse(WANDB_URL)
    candidates = {p.netloc, p.hostname}
    try:
        import netrc
        hosts = netrc.netrc().hosts
        return any(c in hosts for c in candidates)
    except Exception:
        return False


def ensure_wandb(verbose: bool = True) -> bool:
    """Return True if wandb logging is ready; False (and continue) otherwise."""
    def log(m):
        if verbose:
            print(m)

    if os.environ.get("WANDB_MODE") == "disabled" or \
            os.environ.get("WANDB_DISABLED") == "true":
        log("[wandb] disabled via env — skipping")
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        log("[wandb] python package not installed — skipping")
        return False
    if shutil.which("docker") is None:
        log("[wandb] docker CLI not found (Docker Desktop integration?) — skipping")
        return False
    try:
        start_server()
    except Exception as e:
        log(f"[wandb] could not start server: {e} — continuing without wandb")
        return False
    if not wait_healthy():
        log(f"[wandb] server not healthy at {WANDB_URL} — continuing without wandb")
        return False

    os.environ["WANDB_BASE_URL"] = WANDB_URL
    os.environ.setdefault("WANDB_PROJECT", PROJECT)

    if not is_logged_in():
        log(f"[wandb] server UP at {WANDB_URL}, but not logged in. One-time setup:")
        log(f"[wandb]   1) open {WANDB_URL} and sign up (creates a local account)")
        log(f"[wandb]   2) copy your API key from {WANDB_URL}/authorize")
        log(f"[wandb]   3) run:  wandb login --host={WANDB_URL} <YOUR_KEY>")
        log("[wandb] continuing WITHOUT wandb until then.")
        return False

    try:
        from ultralytics import settings
        if not settings.get("wandb"):
            settings.update({"wandb": True})
    except Exception as e:
        log(f"[wandb] couldn't enable Ultralytics integration: {e}")
        return False

    log(f"[wandb] ready -> {WANDB_URL}  (project: {os.environ['WANDB_PROJECT']})")
    return True


if __name__ == "__main__":
    ok = ensure_wandb()
    raise SystemExit(0 if ok else 1)
