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
| `n_test` | Test subjects with measured retinotopy | -1 (all, 178) | 178 | More statistical power; 1.4 GB and a few CPU minutes per subject |
| `n_train`, `n_val` | Training and validation subjects | 30, 8 | about 900 available | Better networks; 1.4 GB each; all training volumes sit in GPU memory |
| `n_7t` | Test subjects also scored on their 7T scan | 40 | `n_test` | 1.1 GB each |
| `n_retest` | Test-retest subjects | 20 | about 20 | Tighter reproducibility interval |
| `n_nsd` | NSD subjects | 8 | 8 | 4 GB per subject per run |
| `train_seeds` | Seeds for the `path` and `voxel` networks | `[0, 1, 2]` | none | One more run of each per seed |
| `variants` | Networks to train | 6 | `path, voxel, onestep, noanat, wmseeds, mlp, fodrefine` | One training run each; `fodrefine` also needs a CSD fit per training subject and protocol |

`n_test`, `n_train`, `n_7t`, `n_retest` and `n_nsd` accept `-1`: every subject that qualifies is taken, and the
sizes of the lists are printed when they are drawn. `n_val` is always a number. Validation subjects are drawn before training subjects, so they stay the same when
`n_train` changes, and a larger `n_train` keeps every subject of a smaller one: runs with different `n_train`
(for example 30, 100, 200) can be compared directly. Each needs its own `outputs/subjects.json`.
| `epochs`, `steps_per_epoch` | Training length per run, `epochs × steps_per_epoch` steps | 60, 500 | none | Time linear; a run started with another length must be deleted and retrained |
| `max_train_hours` | Optional wall-clock limit per run | `null` (off) | none | A limit gives slower variants fewer steps; keep `null` for reported results |
| `seeds_per_vertex` | Tractography seeds per V1–V3 vertex | 10 | none | More streamlines, linear tracking time |
| `n_null` | Surrogates per streamline | 100 | none; below 20 is noisy | Smoother null, linear time and memory |
| `n_jobs` | CPU workers for the per-session steps | -2 (all cores but one) | a number, or negative to count back from all cores | Faster; about 2 GB of memory per worker |
| `gpu_jobs` | Sessions at once in network inference (notebooks 06, 07) | 4 | 1 or more | File reading and writing overlap with GPU work; results do not depend on it |
| `run_protocols` | Protocols evaluated | 9 | names under `protocols` | One more column of the grid |

Every run trains the same 30,000 steps. Each log row in `outputs/models/<run>/log.json` carries the hours so far,
so the first epoch shows how long a run will take. With the defaults there are ten runs:
`path` and `voxel` with three seeds each, plus `onestep`, `noanat`, `mlp` and `fodrefine`.

After changing any `n_*`, delete `outputs/subjects.json` so the lists are drawn again; the notebooks stop with a
message when the file does not match `n_test`, `n_train` or `n_val`.

To compare training sizes on one machine, run the experiment once per size, each with its own output folder:
set `n_train` and `paths.out` (for example `n_train: 50`, `out: outputs_train50`) and run `run_all.sh full`. The
runs share `data/` (crops, reference and classical fODFs, their tractograms and scores), so a second run only
trains its networks and scores them; it keeps its own `subjects.json`, networks, results and figures.

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
| Cached | `seed`, `hemis`, `protocols`, `n_b0`, `crop_margin_mm`, `ref_voxel`, `sh_order_*`, `shells_in`, `ref_model`, `wm_mask_thr`, `rumba_iters`, `tracking.*`, `seeds_per_vertex`, `nsd.seeds_per_vertex`, `nsd.flip_x` | Delete `data/cache` (or `data/cache/dev`). The code stops and names the key otherwise |
| Scoring | `prf.*`, `n_null`, `stats.*`, `nsd.ecc_max` | Nothing; scores are recomputed from cached tractograms |
| Network | `model.*`, `patch`, `base_channels`, `train_protocols` | Delete the run folders in `outputs/models/`. A run that starts from scratch removes the cached outputs of the earlier network with the same name |

`sh_order_in`, `sh_order_out` and `shells_in` set the network's input and output size, so changing them needs both.
