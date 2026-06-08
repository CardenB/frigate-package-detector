#!/usr/bin/env bash
# ============================================================================
# One-shot interactive setup. Paste your Roboflow API key when prompted and
# this will:
#   1. write it into .env
#   2. create the venv + install torch(cu128) + deps (via setup.sh)
#   3. optionally download all datasets, build the merged set, and report balance
#
# Safe to re-run: it skips the venv build if .venv already exists, and updates
# the key in place without duplicating lines.
#
# Usage:
#   ./bootstrap.sh                 # interactive
#   ./bootstrap.sh --reinstall     # rebuild the venv from scratch
#   ROBOFLOW_API_KEY=xxx ./bootstrap.sh --yes   # non-interactive (CI/headless)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

REINSTALL=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --reinstall) REINSTALL=1 ;;
    --yes|-y)    ASSUME_YES=1 ;;
    -h|--help)   grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1;36m>> %s\033[0m\n' "$*"; }

# Rewrites KEY=VALUE in .env safely (no sed delimiter issues with the key chars).
update_env() {
  local key="$1" val="$2"
  if grep -q "^${key}=" .env 2>/dev/null; then
    VAL="$val" awk -v k="$key" \
      'BEGIN{v=ENVIRON["VAL"]} $0 ~ "^"k"=" {print k"="v; next} {print}' \
      .env > .env.tmp && mv .env.tmp .env
  else
    printf '%s=%s\n' "$key" "$val" >> .env
  fi
}

# ---- 1. .env + Roboflow key -------------------------------------------------
[[ -f .env ]] || { cp .env.example .env; say "Created .env from .env.example"; }

RF_KEY="${ROBOFLOW_API_KEY:-}"
if [[ -z "$RF_KEY" && "$ASSUME_YES" -eq 0 ]]; then
  echo
  echo "Roboflow API key — free at https://app.roboflow.com/settings/api"
  echo "(press Enter to skip Roboflow and use only the no-account sources)"
  read -rsp "Paste Roboflow API key: " RF_KEY
  echo
fi

if [[ -n "$RF_KEY" ]]; then
  update_env ROBOFLOW_API_KEY "$RF_KEY"
  say "Saved ROBOFLOW_API_KEY to .env"
else
  say "No key provided — Roboflow sources will be skipped (others still run)"
fi

# ---- 2. venv + deps ---------------------------------------------------------
if [[ "$REINSTALL" -eq 1 ]]; then
  say "Removing existing .venv (--reinstall)"; rm -rf .venv
fi

if [[ ! -d .venv ]]; then
  say "Building venv + installing torch(cu128) + deps"
  ./setup.sh
else
  say "Reusing existing .venv (pass --reinstall to rebuild)"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
say "Active python: $(command -v python)"

# ---- 3. optional: run the data pipeline now ---------------------------------
RUN_DATA="$ASSUME_YES"
if [[ "$ASSUME_YES" -eq 0 ]]; then
  echo
  echo "Download datasets + build the merged set now?"
  echo "(this pulls several GB from Roboflow/Open Images/COCO; can take a while)"
  read -rp "Run 'make data && make analyze' now? [y/N]: " ans
  [[ "$ans" =~ ^[Yy] ]] && RUN_DATA=1
fi

if [[ "$RUN_DATA" -eq 1 ]]; then
  say "make data  (download -> convert -> build)"
  make data
  say "make analyze  (class balance)"
  make analyze
  cat <<'NEXT'

Dataset is built. Review the balance report above, then train + export:
    source .venv/bin/activate
    make train
    make export
Then wire it into Frigate per frigate/config.snippet.yaml
NEXT
else
  cat <<'NEXT'

Setup complete. When ready:
    source .venv/bin/activate
    make data && make analyze     # build + inspect the dataset
    make train && make export     # finetune + export ONNX for Frigate
NEXT
fi
