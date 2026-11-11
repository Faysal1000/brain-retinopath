# Recovering retinotopic V1–V2–V3 connectivity from clinical-grade diffusion MRI with path-consistent deep learning

Short association fibres between visual areas V1, V2 and V3 link cortical points that represent the same location
of the visual field. High-quality diffusion MRI can show this organisation; clinical-grade scans mostly cannot.
This repository tests whether a network that enhances fibre orientation estimates from clinical-grade scans can
bring it back, and measures the answer against each person's own measured retinotopy.

The code name is `retinopath`. Background, method and design are in [docs/overview.md](docs/overview.md).

## Layout

| Path | Contents |
|---|---|
| `config.yaml` | All paths and settings, each documented in place |
| `notebooks/` | The experiment, `00` to `09`, run in order |
| `src/retinopath/` | Library used by the notebooks |
| `scripts/` | Setup, data download, running the notebooks |
| `AGENTS.md` | Rules for changing the code and keeping the docs in step |
| `docs/` | Documentation (list below) |
| `data/`, `outputs/` | Created at run time, not tracked |

## Run on macOS (Apple silicon)

Uses the MPS device. Intended for the small test configuration (`dev:` block of `config.yaml`: a few subjects,
one hemisphere, short training).

```bash
bash scripts/setup.sh
```
```bash
cp .env.example .env    # then add your HCP keys, see docs/data.md
```
```bash
bash scripts/download_data.sh dev
```
```bash
bash scripts/run_all.sh dev
```

## Run on NVIDIA DGX Spark (Linux aarch64, CUDA 13)

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip python3-dev git tmux
```
```bash
git clone <repository-url> retinopath && cd retinopath
```
```bash
bash scripts/setup.sh
```
```bash
cp .env.example .env    # then add your HCP keys, see docs/data.md
```
```bash
bash scripts/download_data.sh
```
```bash
bash scripts/run_all.sh full
```

`run_all.sh` executes the ten notebooks in order and saves executed copies to `outputs/executed/`. It can be
stopped at any time; running the same command again continues from where it stopped. The notebooks can also be
opened in Jupyter with the `retinopath` kernel and run by hand (`DEV_MODE` is set in the first cell).

## Documentation

- [docs/overview.md](docs/overview.md): the question, what we do, hypotheses, pipeline and network diagrams, notebooks
- [docs/data.md](docs/data.md): datasets, access, download, terms of use
- [docs/preprocessing.md](docs/preprocessing.md): crops, tissue maps, retinotopy, protocol simulation, network inputs, classical fODFs
- [docs/network.md](docs/network.md): architectures, path-consistency loss, variants, training
- [docs/scoring.md](docs/scoring.md): tractography, retinotopic scores, statistics
- [docs/code.md](docs/code.md): files, modules, cached steps, outputs
- [docs/configuration.md](docs/configuration.md): settings and what changing them costs
- [docs/running.md](docs/running.md): setup, stopping and resuming, GPUs, troubleshooting

Rules for changing the code are in [AGENTS.md](AGENTS.md).

## Data terms

The Human Connectome Project and Natural Scenes Dataset data are not redistributed here and must not be committed.
See [docs/data.md](docs/data.md) for the terms and the required acknowledgements.
