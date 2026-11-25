# Configuration

Everything is set in `config.yaml`; each key is commented there with its unit, range and cost. This page gives the
structure and the rules for changing things.

## Two sizes

`DEV_MODE` in the first cell of each notebook (or `dev` / `full` given to `scripts/run_all.sh`) selects a block:

- `full:` the experiment.
- `dev:` a small run on a few named subjects, one hemisphere, short training. Its cache and outputs are kept apart
  in `data/cache/dev` and `outputs/dev`.

The chosen block is merged over the top-level keys. Both blocks hold the same keys.

## Size of the experiment

| Key | Meaning | Default | Limit | Raising it |
|---|---|---|---|---|
| `n_test` | Test subjects with measured retinotopy | -1 (all: 178, about 164 without `development_subjects`) | 178 | More statistical power; 1.4 GB and a few CPU minutes per subject |
| `n_train`, `n_val` | Training and validation subjects | 12, 8 | about 900 available | Better networks; 1.4 GB each; all training volumes sit in GPU memory |
| `n_7t` | Test subjects also scored on their 7T scan | 40 | `n_test` (most test subjects have a 7T scan) | 1.1 GB each |
| `n_retest` | Test-retest subjects | 20 | about 20 | Tighter reproducibility interval |
| `n_nsd` | NSD subjects | 8 | 8 | 4 GB per subject per run |
| `train_seeds` | Seeds for the `path` and `voxel` networks | `[0, 1, 2]` | none | One more run of each per seed |
| `variants` | Networks to train | 6 | `path, voxel, onestep, noanat, wmseeds, signal, mlp` | One training run each; the fODF-input variants need a CSD fit per training subject and protocol (notebook 02) |
| `epochs`, `steps_per_epoch` | Training length per run, `epochs × steps_per_epoch` steps | 60, 500 | none | Time linear; raise if the validation curves of notebook 05 still fall at the end; a run started with another length must be deleted and retrained |
| `max_train_hours` | Optional wall-clock limit per run | `null` (off) | none | A limit gives slower variants fewer steps; keep `null` for reported results |
| `seeds_per_vertex` | Tractography seeds per V1–V3 vertex (about 2,000 vertices per subject on the HCP mesh) | 10 | none | More stable scores, linear tracking time |
| `batch_size` | Training batch | 16 | an integer or `auto` | Time per step about linear (about 2.6 s per step at 64 on a DGX Spark); use one value for all runs |
| `patch` | Cubic training patch (voxels) | 48 | multiple of 2^(`levels`−1); at least 40, since the 40-step path covers 25 mm = 20 voxels | More context, more GPU memory and time; smaller volumes are padded |
| `base_channels` | U-Net width | 48 | multiple of 8 | Memory and time grow about quadratically |
| `train_protocols` | Network inputs in training | P1, P2, P3, D32, T7 | names under `protocols` or `extra_protocols` | One more cached input per training subject; T7 (b = 1000 and 2000, 1.25 mm) is similar to the HCP 7T scan |
| `keep_raw` | Keep the raw HCP diffusion folder after cropping | `true` | | `false` saves about 1.3 GB per subject; a re-download is then needed if the crop settings change |
| `n_null` | Surrogates per streamline | 20 | below 20 is noisy | Smoother null, linear time and memory; a streamline with fewer distance-matched vertices than `n_null` is left out, and on the fsLR mesh few have more than 20 (none have 100) |
| `n_jobs` | CPU workers for the per-session steps | -2 (all cores but one) | a number, or negative to count back from all cores | Faster; about 2 GB of memory per worker |
| `gpu_jobs` | Sessions at once in network inference (notebooks 06, 07) | 4 | 1 or more | File reading and writing overlap with GPU work; results do not depend on it |
| `run_protocols` | Protocols evaluated | 9 | names under `protocols` | One more column of the grid |

`n_test`, `n_train`, `n_7t`, `n_retest` and `n_nsd` accept `-1`: every subject that qualifies is taken, and the
sizes of the lists are printed when they are drawn. `n_val` is always a number. Validation subjects are drawn
before training subjects, so they stay the same when `n_train` changes, and a larger `n_train` keeps every subject
of a smaller one: runs with different `n_train` (12, 35, 100) can be compared directly. Each needs its own
`subjects.json`.

`development_subjects` lists the subjects with retinotopy that were looked at while the method and its settings
were chosen: the two dev test subjects and the 12 test subjects of the development runs. They never enter a test
list of the full run (`test`, and with it `t7`, and `retest`), so the reported results come only from subjects
that played no part in choosing the method.
Having retinotopy, they are not in the training pool either. What was measured on them is in
[09-development.md](09-development.md).

Every run trains the same 30,000 steps. Each log row in `outputs/models/<run>/log.json` carries the hours so far,
so the first epoch shows how long a run will take. With the defaults there are ten runs:
`path` and `voxel` with three seeds each, plus `onestep`, `noanat`, `signal` and `mlp`.

After changing any `n_*`, delete `outputs/subjects.json` so the lists are drawn again; the notebooks stop with a
message when the file does not match `n_test`, `n_train` or `n_val`.

## Training size

The experiment is run with 12 training subjects (the confirmatory analysis: a paired set of this size is what one
site can collect) and repeated with 35 and 100. Each further size is one more run with its own output folder: set
`n_train` and `paths.out` (`n_train: 35`, `out: outputs_train35`; then `n_train: 100`, `out: outputs_train100`)
and run `run_all.sh full`. These runs need only the two main networks (`variants: [path, voxel]`); the ablations
belong to the 12-subject run. The runs share `data/` (crops, reference and classical fODFs, their tractograms and
scores), so a further run only trains its networks and scores them; it keeps its own `subjects.json`, networks,
network outputs, results and figures. `training_size_runs` names these folders. With `n_train` and `paths.out`
set back, running notebooks 09 and 10 again collects the tests of every size
([07-scoring.md](07-scoring.md#training-size-training_size_testscsv)).

## Protocols

```yaml
P2: {shells: {1000: 32}, voxel: 2.0}          # b-value: number of directions; voxel size in mm
P3: {shells: {1000: 20}, voxel: 2.5, snr: 15} # snr adds Rician noise at this b=0 white-matter SNR
```

`all` keeps every direction of a shell (90 in HCP 3T). Voxel size must be at least 1.25 mm. `REF` is the full
acquisition. `clinical_protocols` are the ones used for the confirmatory tests and for the networks.
`extra_protocols` are used for training only.

## What needs clearing after a change

| Group | Keys | After a change |
|---|---|---|
| Cached | `seed`, `hemis`, `protocols`, `n_b0`, `crop_margin_mm`, `ref_voxel`, `sh_order_*`, `shells_in`, `ref_model`, `wm_mask_thr`, `rumba_iters`, `nsd.flip_x` | Delete `data/cache` (or `data/cache/dev`). The code stops and names the key otherwise |
| Tracking | `tracking.*`, `seeds_per_vertex`, `nsd.seeds_per_vertex` | Nothing; each tractogram records its settings and is tracked again when they differ |
| Scoring | `prf.*`, `n_null`, `stats.*`, `nsd.ecc_max` | Nothing; scores are recomputed from cached tractograms |
| Network | `model.*`, `patch`, `base_channels`, `train_protocols` | Delete the run folders in `outputs/models/`. A run that starts from scratch removes the cached outputs of the earlier network with the same name |

`sh_order_in`, `sh_order_out` and `shells_in` set the network's input and output size, so changing them needs both.

Other trade-offs: a smaller `crop_margin_mm` cuts long streamlines, a larger one costs memory and time roughly
cubically; `ref_voxel` is the HCP 3T voxel size and should stay 1.25 mm; a larger `tracking.endpoint_tol_mm`
accepts more streamlines with less precise endpoints; a smaller `tracking.step_mm` is slower and smoother; time is
linear in `rumba_iters`, `tracking.particles` and `stats.n_boot`.
