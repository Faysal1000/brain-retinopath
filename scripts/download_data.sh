#!/usr/bin/env bash
# Download every file a run needs, before running any notebook.
#   bash scripts/download_data.sh              everything for the `full:` block of config.yaml
#   bash scripts/download_data.sh dev          the few subjects of the `dev:` block
#   WORKERS=32 bash scripts/download_data.sh   more files at once (default 16)
# Needs .env with the HCP keys. Complete files are skipped: stop with Ctrl-C and run again to continue.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || { echo "run setup first: bash scripts/setup.sh"; exit 1; }
[ -f .env ] || { echo ".env missing: cp .env.example .env and add your HCP keys"; exit 1; }
FLAG=""; [ "${1:-full}" = "dev" ] && FLAG="--dev"
PYTHONWARNINGS=ignore .venv/bin/python -m retinopath.pipeline $FLAG --workers "${WORKERS:-16}"
du -sh data/hcp data/nsd data/osf 2>/dev/null || true
