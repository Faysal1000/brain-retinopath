#!/bin/sh
# Run one notebook in place and print its text outputs. Usage: scripts/run_notebook.sh 03_retinotopy_metric
cd "$(dirname "$0")/.." || exit 1
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=retinopath "notebooks/$1.ipynb" 2>&1 | tail -25
