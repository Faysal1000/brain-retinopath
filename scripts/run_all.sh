#!/usr/bin/env bash
# Run every notebook in order, without a browser.  Usage:  bash scripts/run_all.sh dev    (small test run)
#                                                          bash scripts/run_all.sh full   (full experiment)
# Stop any time with Ctrl-C. Run the same command again later: it continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  dev)  FLAG=True ;;
  full) FLAG=False ;;
  *) echo "usage: bash scripts/run_all.sh dev|full"; exit 1 ;;
esac
mkdir -p outputs/executed
for nb in notebooks/0*.ipynb; do
  name=$(basename "$nb" .ipynb)
  echo "=== $name  ($(date '+%F %T'))"
  # set the DEV_MODE flag for this run in a copy; the notebooks in notebooks/ are not modified
  .venv/bin/python - "$nb" "outputs/executed/$name.ipynb" "$FLAG" <<'PY'
import sys, re, nbformat
nb = nbformat.read(sys.argv[1], 4)
c = nb.cells[1]; c.source = re.sub(r"^DEV_MODE = (True|False)", f"DEV_MODE = {sys.argv[3]}", c.source, count=1)
assert f"DEV_MODE = {sys.argv[3]}" in c.source
nbformat.write(nb, sys.argv[2])
PY
  PYTHONWARNINGS=ignore .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=retinopath \
    --ExecutePreprocessor.cwd=notebooks "outputs/executed/$name.ipynb" 2>&1 | tail -3
done
echo "=== all notebooks finished. Executed copies with their outputs: outputs/executed/"
