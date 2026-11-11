# Running

Two platforms are supported: macOS on Apple silicon (MPS) for the small `dev` configuration, and Linux with CUDA
(NVIDIA DGX Spark, aarch64, CUDA 13) for the full experiment. The device is chosen automatically: CUDA, then MPS,
then CPU.

## Setup

`scripts/setup.sh` creates `.venv`, installs the packages, registers the `retinopath` Jupyter kernel and runs a
short device check. It needs Python 3.12 or 3.13; package versions are pinned in the requirements files.

**macOS**

```bash
brew install python@3.12      # only if python3 is missing or older than 3.12
bash scripts/setup.sh
```

**DGX Spark**

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip python3-dev git tmux
bash scripts/setup.sh
```

On Linux the script installs PyTorch from the CUDA 13 wheel index (`cu130`), which carries its own CUDA
libraries; the CUDA toolkit does not need to be installed. `nvidia-smi` must list the GPU. If the system Python is
older than 3.12:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
PYTHON=python3.12 bash scripts/setup.sh
```

Then create `.env` with the HCP keys ([02-data.md](02-data.md)).

## Download, then run

```bash
bash scripts/download_data.sh [dev]      # all data for the chosen configuration
bash scripts/run_all.sh dev|full         # notebooks 00 to 10 in order
```

`run_all.sh` copies each notebook to `outputs/executed/`, sets `DEV_MODE` there and executes it, so the notebooks
in `notebooks/` stay clean. To work interactively instead:

```bash
.venv/bin/jupyter lab --no-browser --ip 0.0.0.0 --port 8888
```

Open the notebooks in order, choose the `retinopath` kernel and set `DEV_MODE` in the first cell.
A single notebook can be executed with `bash scripts/run_notebook.sh 03_retinotopy_metric`.

On a remote machine, start long runs inside `tmux` (`tmux new -s run`; detach with Ctrl-b d; return with
`tmux attach -t run`).

## Stopping and resuming

A run can be stopped at any point with Ctrl-C or a kernel interrupt. Running the same command or notebook again
continues from where it stopped.

| Stage | On restart |
|---|---|
| Downloads | Complete files are skipped |
| Crops, fODFs, tractograms, network outputs | One file per session, protocol and method; existing files are reused |
| Scores | One file per session, protocol, method and analysis setting; existing files are read back |
| Training | Each run saves a checkpoint when interrupted and every `model.ckpt_minutes`, and resumes from it. Finished runs are skipped. Steps and hours already trained are kept in the checkpoint. A run whose notebook or shell was killed saves and exits by itself; a run that is still alive when the same run is started again is told to save and stop first |

Files are written under a temporary name and renamed when complete, so an interrupted write leaves nothing behind.
After a power loss a training run loses at most the last checkpoint interval. To redo a step, delete its file in
`data/cache/<session>/` or the run folder in `outputs/models/`.

## Outputs

```
outputs/subjects.json     subject lists
outputs/models/<run>/     checkpoint and training log of each network
outputs/logs/<run>.log    console output of each training run
outputs/results/*.csv     result rows per subject, protocol and method; tests and summary tables
outputs/figures/          figures
outputs/run_log.csv       wall-clock hours per stage
outputs/executed/         executed notebooks
```

The main result columns are listed in [08-code.md](08-code.md#result-columns).

## GPUs and nodes

- **One GPU.** Training keeps all volumes on the GPU, uses mixed precision and, with `batch_size: auto`, the
  largest batch that fits (up to 64). Each epoch's line in `outputs/logs/<run>.log` gives the hours left for that
  run and the peak GPU memory (`gpu_gb`, and `gpu_reserved_gb` held by PyTorch). Work per run grows with
  `batch_size × epochs × steps_per_epoch`, so a fixed smaller `batch_size` or fewer `epochs` shortens training;
  use the same values for every network. fODF fitting uses all CPU cores but one (`n_jobs`); tracking and scoring run one
  session per core.
- **Several GPUs in one machine.** `multi_gpu: jobs` trains one network per GPU at the same time.
  `multi_gpu: ddp` makes each training run use all GPUs with PyTorch distributed training; the learning rate is
  scaled by the number of processes.
- **Several machines.** List their host names under `nodes:`. This needs passwordless SSH from the machine that
  runs the notebook and the project at the same path on every node with shared `data/` and `outputs/`. Training
  runs are distributed over the nodes; the other stages run locally. An interrupt does not reach remote runs, which
  rely on their periodic checkpoint.

## Troubleshooting

**PyTorch does not see the GPU (Linux).**
`.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`
should print a `+cu130` build, `13.0` and `True`. If the build is `+cpu`:

```bash
.venv/bin/pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130
```

Another wheel index can be given with `TORCH_INDEX=<url> bash scripts/setup.sh`.

**"config.yaml changed in [...]".** A cached setting was changed; delete `data/cache` (or `data/cache/dev`) or
restore the value ([04-configuration.md](04-configuration.md)).

**HCP download fails with an access error.** Check the two keys in `.env`; they are tied to the account that
accepted the HCP terms.

**Out of GPU memory in training.** Set `batch_size` to a number, or lower `patch` or `base_channels`.

**macOS crash in native code.** PyTorch and DIPY each load an OpenMP runtime. PyTorch CPU threads are limited to
one on macOS (`core.py`), and the probabilistic tracker runs in a separate process there (`track.py`); keep both.
