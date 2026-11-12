#!/usr/bin/env bash
# Run every notebook in order, without a browser.  Usage:  bash scripts/run_all.sh dev    (small test run)
#                                                          bash scripts/run_all.sh full   (full experiment)
# Stop any time with Ctrl-C. Run the same command again later: it continues where it stopped.
# WORKERS=16 bash scripts/run_all.sh full   uses more parallel downloads (default 8).
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  dev)  FLAG=True ;;
  full) FLAG=False ;;
  *) echo "usage: bash scripts/run_all.sh dev|full"; exit 1 ;;
esac
mkdir -p outputs/executed
# Download first, with WORKERS parallel downloads; notebook 01 then only checks the files.
echo "=== download  ($(date '+%F %T'))"
DFLAG=""; [ "$1" = "dev" ] && DFLAG="--dev"
PYTHONWARNINGS=ignore .venv/bin/python -m retinopath.pipeline $DFLAG --workers "${WORKERS:-8}"
for nb in notebooks/[0-9][0-9]_*.ipynb; do
  name=$(basename "$nb" .ipynb)
  echo "=== $name  ($(date '+%F %T'))"
  # runs a copy with DEV_MODE set and prints its output live; the notebooks in notebooks/ are not modified
  PYTHONWARNINGS=ignore .venv/bin/python scripts/run_notebook.py "$nb" "outputs/executed/$name.ipynb" --dev "$FLAG"
done
echo "=== all notebooks finished. Executed copies with their outputs: outputs/executed/"
