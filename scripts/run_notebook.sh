#!/bin/sh
# Run one notebook, printing its output live, with the DEV_MODE saved in it. The executed copy goes to
# outputs/executed/; the notebook itself stays without outputs.  Usage: scripts/run_notebook.sh 03_retinotopy_metric
cd "$(dirname "$0")/.." || exit 1
PYTHONWARNINGS=ignore .venv/bin/python scripts/run_notebook.py "notebooks/$1.ipynb" "outputs/executed/$1.ipynb"
