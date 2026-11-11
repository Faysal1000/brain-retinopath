# AGENTS.md

Guide for anyone, person or coding agent, who changes this repository.

## What this project is

`retinopath` tests whether deep learning can recover the retinotopic organisation of short V1–V2–V3 fibres from
clinical-grade diffusion MRI. Each tractogram is scored against the subject's own measured retinotopy: a
streamline between two visual areas should join points that see the same location of the visual field. The
retinotopic specificity index (RSI) compares each streamline's endpoint error with distance-matched surrogates.

- **H1:** on full-quality HCP data the reference tractogram is retinotopic (RSI > 0).
- **H2:** a network trained voxel by voxel recovers more of the lost voxel-wise accuracy than of the lost RSI.
- **H3 (main result):** the same network trained with an added path-consistency loss (a differentiable tracker
  through the predicted and the reference fODF) has higher RSI on clinical-grade protocols, without lower
  voxel-wise accuracy. The decisive comparison is `path` vs `voxel`, two networks that differ only in the loss.

Data: HCP Young Adult (3T, 7T, retest) and the Natural Scenes Dataset, all public. Details:
[docs/01-overview.md](docs/01-overview.md); the README lists all documentation.

## Layout

`config.yaml` holds every setting. `notebooks/00`–`09` are the experiment and call the library in
`src/retinopath/`. `scripts/` sets up the environment, downloads data and runs the notebooks. Every file is listed
in [docs/08-code.md](docs/08-code.md).

## Rules

**Data and privacy**
- Public data only. Never commit `data/`, `outputs/`, raw scans or any per-subject derived file; the HCP and NSD
  terms forbid redistribution.
- The HCP keys live only in `.env`. Never print, log or copy them.
- Never create accounts or accept data terms on someone's behalf.
- `PROPOSAL.md`, `RESEARCH_NOTES.md` and `paper/` are private and git-ignored. `paper/` holds an unpublished
  manuscript: do not copy its text into the repository and do not send it anywhere.

**Science**
- Never invent numbers or citations. A reference is added only after opening the source (DOI, journal page).
  Numbers in documents come from files in `outputs/` of a full run, never from dev runs.
- No test subject (anyone with measured retinotopy) may enter training or validation. Splits are by subject and
  drawn once with the seed (`outputs/subjects.json`).
- Methods are compared under identical conditions: same seeds, tracking grid, tissue maps, tracker settings and
  scoring. Networks are trained for the same number of steps (`max_train_hours: null`). Keep it that way when
  adding a method.
- Analysis choices (thresholds, eccentricity range, margins) are fixed in `config.yaml` before test scores are
  looked at; alternatives belong in notebook 08 as robustness checks.

**Code**
- Every step must be resumable: write results with `core.atomic`, skip work whose file exists, and make training
  checkpoint and resume. A run that is stopped and started again must give the same result.
- A setting that changes cached files must be in `core.CACHED_KEYS`; a setting that only changes scores must be in
  the score hash in `pipeline.scores`.
- Fixed seeds everywhere (`cfg.seed`); no unseeded randomness.
- No hard-coded run-time limits. Sizes and lengths come from `config.yaml`, each key commented there with its
  unit, range and cost.
- The full experiment runs on Linux aarch64 with CUDA 13 (NVIDIA DGX Spark); macOS (MPS) is for `dev` runs.
  Every dependency must have Linux aarch64 wheels. Versions are pinned in both requirements files; change them
  together.
- Style: ordinary research code. Short docstrings that say what a function returns and why a choice was made;
  comments only where the code does not explain itself. No status notes, to-do lists, "not implemented" remarks
  or schedules in code, comments, docs or `config.yaml`.
- Keep notebooks free of outputs. Executed copies go to `outputs/executed/` through `scripts/run_all.sh`.
- Do not commit or push unless asked.

## Checking a change

```bash
bash scripts/run_all.sh dev
```

runs all ten notebooks on the `dev:` subjects (a few minutes of training; its outputs in `outputs/dev/` are not
results). A single notebook: `bash scripts/run_notebook.sh 03_retinotopy_metric`. A cached step that should be
redone needs its file deleted from `data/cache/dev/<session>/`. Never run the full configuration on a laptop.

## Keep the docs in step with the code

When a change alters behaviour, update the pages that describe it in the same change. README stays short (what
it is, layout, how to run); details go in `docs/`.

| Change in | Update |
|---|---|
| `config.yaml` keys, defaults or what needs clearing | comment in `config.yaml`, [docs/04-configuration.md](docs/04-configuration.md) |
| `subject.py`, `nsd.py`, `atlas.py`, `degrade.py`, `recon.py` | [docs/05-preprocessing.md](docs/05-preprocessing.md) |
| `model.py`, `train.py`, `jobs.py` | [docs/06-network.md](docs/06-network.md) |
| `track.py`, `metrics.py`, `stats.py` | [docs/07-scoring.md](docs/07-scoring.md) |
| `pipeline.py`, `core.py`, `download.py`, files, cache files, result columns | [docs/08-code.md](docs/08-code.md) |
| A notebook, the hypotheses or the methods compared | [docs/01-overview.md](docs/01-overview.md) |
| Datasets, downloaded files, terms | [docs/02-data.md](docs/02-data.md) |
| Setup, scripts, platforms, resume behaviour | [docs/03-running.md](docs/03-running.md), README if the commands change |
