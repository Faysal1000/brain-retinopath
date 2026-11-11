# Network and training

The learned methods: architectures, the differentiable tracker behind the path-consistency loss, the losses, how
training batches are drawn, the training loop, running on several GPUs, and inference. Code:
[model.py](../src/retinopath/model.py), [train.py](../src/retinopath/train.py),
[jobs.py](../src/retinopath/jobs.py).

## Task

Input: one degraded scan of a training subject (any of `train_protocols`: P1, P2, P3, D32, T7) on the 1.25 mm crop
grid ([05-preprocessing.md](05-preprocessing.md#network-input-reconsignal_sh-pipelinenet_input)). Target: the reference
fODF of the same subject (multi-shell CSD on the full data, 45 SH coefficients). The network never sees
retinotopy, and no subject with retinotopy is used for training or validation.

## Architectures (`model.py`)

**`UNet3D`** (main network). A residual 3D U-Net with `levels` (3) resolution levels and `base_channels` (48)
channels at the top, doubling per level (48, 96, 192).

- `Block`: two 3×3×3 convolutions, each followed by GroupNorm (8 groups, 4 if the width is not a multiple of 8)
  and SiLU, plus a 1×1×1 convolution on the skip path when the width changes.
- Down: one block per level, 2× average pooling between levels.
- Up: trilinear upsampling to the size of the skip, concatenation with the skip, one block.
- Head: 1×1×1 convolution to 45 output channels.
- Input: 45 signal channels + 7 anatomical channels = 52 (45 for `noanat`). About 3.3 million parameters.

The architecture is kept plain on purpose: the variants that test the loss differ only in the loss.

**`VoxelMLP`** (`mlp`). Four 1×1×1 convolutions of width 256 with SiLU and an output layer: a per-voxel network
without spatial context, in the spirit of Karimi et al. (2021). About 0.22 million parameters.

**`FODRefine`** (`fodrefine`). The input is the classical CSD fODF of the degraded scan (45 channels); the output
is `input + UNet3D(input)`. The head of the U-Net is initialised to zero, so the network starts as the identity
(it returns CSD) and learns a correction. This follows the idea of FOD-Net (Zeng et al. 2022) with the same U-Net
as the other variants, so only the input differs.

`build_model(cfg, kind)` builds `unet`, `mlp` or `refine` with the channel counts from `config.yaml`.

## Soft tracker (`PathTracer`)

A deterministic tracker whose positions are differentiable with respect to the fODF. All coordinates are voxel
indices of the patch.

1. **Amplitudes** (`amps`). The fODF is sampled trilinearly at each point (`sample_trilinear`, written with
   `gather` so it runs on CUDA, MPS and CPU; points outside the volume are clamped to the border) and evaluated on
   the 362 directions of `symmetric362`. Negative amplitudes are set to 0.
2. **Start direction** (`start_dirs`). The sphere direction maximising amplitude × relu(cos to the inward
   normal)². A hard choice, taken from the reference fODF and shared by the predicted and the reference path.
3. **Step direction** (`step_dir`). Weights w = (amplitude / max amplitude)^`sharpen` (4) for directions within
   `cone_deg` (45°) of the previous direction, 0 outside. The new direction is the normalised weighted mean of the
   directions: a smooth stand-in for "follow the peak nearest to the current direction". If every weight is zero
   the previous direction is kept.
4. **Trace** (`trace`). Euler steps of `step_mm` (0.625 mm, half a voxel) from the seeds; returns every position.

## Losses (`train.losses`, `model.path_loss`, `model.nonneg_loss`)

    total = voxel + λ_nonneg · nonneg + λ_path · path_mm / (steps · step_mm)

- **Voxel loss.** Mean squared error over the 45 coefficients in voxels with wm > `wm_mask_thr`, divided by
  `scale²`. `scale` is the standard deviation of the reference coefficients in voxels with wm > 0.5 of the
  training volumes, which makes the loss independent of the signal scale. It is the loss of every voxel-wise
  method.
- **Non-negativity penalty.** At 2,000 random voxels of each patch, the negative part of the predicted amplitudes
  on the 362 directions, weighted by the white-matter fraction and divided by `scale`. `lambda_nonneg` = 0.1.
- **Path-consistency loss** (`lambda_path` = 1). From `seeds_per_patch` (64) seeds under the cortex, the soft
  tracker follows the reference fODF (no gradient) and the predicted fODF for `steps` steps with the same start
  directions. A reference point counts while the path is more than one voxel inside the patch and in white matter
  (wm > 0.1 by trilinear sampling); after it leaves, all later points are masked. The loss is the Huber distance
  (δ = 2 voxels per axis, summed over the three axes) between predicted and reference positions, times the voxel
  size (mm), averaged over the counted points; 0 if there are none. Before weighting it is divided by the path
  length `steps · step_mm`, so it is a relative endpoint error. Errors accumulate along the path, which is the
  point: a small direction error early on moves every later position, as it does for a streamline.
- **Curriculum.** `steps` grows linearly from `steps_start` (4) to `steps_end` (40 steps, 25 mm) over the first
  half of training, then stays at 40.

## Variants (`train.VARIANTS`, `apply_variant`)

| Variant | Change from `path` | Question |
|---|---|---|
| `path` | none: voxel + non-negativity + path loss | Proposed method |
| `voxel` | `lambda_path = 0` | The H3 control: same network, voxel loss only |
| `onestep` | Path loss replaced by (1 − cos) between the predicted and the reference step direction at each reference position, times step length × steps / 2 (a similar mm range) | Does accumulation along the path matter? |
| `noanat` | No anatomical channels | Does the anatomy input matter? |
| `wmseeds` | Path seeds at random white-matter voxels (wm > 0.9) with random start directions | Does seeding under the cortex matter? |
| `mlp` | `VoxelMLP`, voxel loss only | A network without spatial context |
| `fodrefine` | `FODRefine`, CSD fODF input, voxel loss only, no anatomy | fODF-to-fODF enhancement |

`path` and `voxel` are trained once per entry of `train_seeds`; the others with the first seed only
(`jobs.train_commands`). A run is called `<variant>_s<seed>`.

## Training batches (`load_volumes`, `sample_batch`)

All training and validation volumes are put on the device once: the network inputs of every training protocol
(float16), the reference fODF and the anatomical channels, each zero-padded so every axis is at least one patch
long. The seed points under V1–V3 (`seed_depth_mm` below each vertex) and their inward normals stay on the CPU,
where patch positions are drawn; only the patches are cut on the device.

For each batch element:

1. A random subject and a random training protocol.
2. A centre: in 70% of cases a random V1–V3 seed point moved by up to ±patch/4 per axis, otherwise anywhere in
   the crop. The cube of `patch` (48) voxels around it is clamped to the volume.
3. Path seeds: 64 of the seed points inside the patch with a 2-voxel margin, drawn with replacement, with their
   inward normals. A patch with no such seed gets dummy seeds marked invalid, which the path loss ignores.

Positions are drawn with a CPU generator seeded from `seed`, the training seed, the process rank and the step at
which the process started, so runs are reproducible and a resumed run does not repeat the batches it has seen.

## Training loop (`train.train`)

- **Optimiser.** AdamW, learning rate `model.lr` (2e-4) × number of processes, weight decay `model.weight_decay`
  (1e-4), gradient-norm clipping at 1.
- **Schedule.** Progress = step / (`epochs` × `steps_per_epoch`), or the fraction of `max_train_hours` if that is
  set and larger. The learning rate follows a cosine from its start value to 0 over the progress, and the
  curriculum uses the same progress. Training ends at progress 1. With the defaults every run is 30,000 steps.
- **Precision.** On CUDA: float16 autocast with a gradient scaler when `model.amp` is true, TF32 matrix maths,
  cuDNN autotuning, and PyTorch's expandable-segment allocator to limit fragmentation. Losses are computed in
  float32.
- **Batch size** (`auto_batch`). An integer `batch_size` is used as given. With `auto` on CUDA, batches of 8, 16,
  32, 64 are tried with one forward and backward pass including the path loss at full length; the largest whose
  peak memory stays under 70% of the device memory is kept (4 if 8 does not fit). A size is not tried if twice the
  working memory of the previous size would exceed the limit, because a GPU that shares memory with the system can
  stall rather than fail on an oversized trial. The trial always includes the path loss, so `path` and `voxel` get
  the same batch. The size is stored in the checkpoint and reused on resume. On MPS and CPU `auto` means 2.
- **Validation.** Every `steps_per_epoch` (500) steps, 8 batches from the validation subjects with a fixed
  generator, scored with the voxel loss and the full-length accumulated path error for every variant, so all
  variants are compared on the same numbers. One row goes to `log.json` and to `outputs/logs/<run>.log`.
- **Checkpoint.** `last.pt` holds the model, optimiser and scaler states, `state` (step, hours, log, `done`) and
  `meta` (variant, seed, kind, batch, world size, scale, total steps). Written every `ckpt_minutes` (15), on
  SIGINT or SIGTERM (the run then exits with code 130) and at the end with `done` set. A finished run is skipped.
  A run whose checkpoint was made for a different total number of steps stops with a message, so all networks
  train equally long; delete its folder to train it again.
- **Fresh start.** A run that starts without a checkpoint deletes cached fODFs, tractograms and scores of an
  earlier network with the same name (`clear_outputs`), so a retrained network is never mixed with results of the
  old one.
- **One process per run** (`take_over`). `outputs/models/<run>/pid` holds the host and process ID training the
  run. If the run is started while an earlier process for it is still alive on the same machine (for example
  left by a killed notebook), the earlier process is sent SIGINT and the new one waits for it to save and exit.
  Processes of the same `torchrun` group are not treated as rivals. A run also saves and stops when its parent
  process (notebook kernel or shell) goes away.
- **Distributed.** Under `torchrun` each GPU runs one process with DistributedDataParallel. Rank 0 broadcasts the
  stop flag and the elapsed hours every step, so all ranks stop at the same step; only rank 0 validates and writes.

## Running training runs (`jobs.py`)

`train_commands` lists one `python -m retinopath.train --variant V --seed S [--dev]` per run. `run_jobs` spreads
them over slots = nodes × GPUs (one slot on a machine without CUDA). Each slot runs its jobs one after another,
with `CUDA_VISIBLE_DEVICES` set to its GPU and, for a remote node, through SSH (same project path on every
node, shared `data/` and `outputs/`). Each run's output goes to `outputs/logs/<run>.log`, and the exit code is
reported (0 finished, 130 stopped with a checkpoint). With `multi_gpu: ddp` on one machine there is one slot and
every run uses all GPUs through `torchrun`. A notebook interrupt sends SIGINT to every local run, waits up to two
minutes for each to save, and stops.

## Inference (`predict`, `net_fodf`, `load_trained`, `trained_runs`)

`load_trained` rebuilds the network of a run from its checkpoint. `predict` passes the whole crop through the
network at once (zero-padded to a multiple of 2^(levels−1) and cropped back), with the same mixed precision as
training. `net_fodf` sets the prediction to zero where wm ≤ `wm_mask_thr` and stores it like a classical fit as
`fodf_<protocol>_net_<variant>_s<seed>.npz`; from then on the network is tracked and scored exactly like CSD.
`trained_runs` returns the runs listed in `config.yaml` whose checkpoint is finished; only these are evaluated.
