# Tractography, scoring and statistics

How each fODF becomes a tractogram, how a tractogram is scored against retinotopy, and how the scores are tested.
Code: [track.py](../src/retinopath/track.py), [metrics.py](../src/retinopath/metrics.py),
[stats.py](../src/retinopath/stats.py), `tract`, `scores` and `retest_scores` in
[pipeline.py](../src/retinopath/pipeline.py), and notebook 09.

## Tractography (`track.py`, `pipeline.tract`)

- **Seeds** (`make_seeds`). `seeds_per_vertex` (10) seeds per V1–V3 white-surface vertex, placed
  `tracking.seed_depth_mm` (1 mm) below the white surface along the normal, with Gaussian jitter of
  `seed_jitter_mm` (0.3 mm). The generator uses `seed`, so every method of a session gets exactly the same seeds.
  NSD uses `nsd.seeds_per_vertex` (1) because its meshes are much denser.
- **Grid.** The fODF is interpolated to the 1.25 mm crop grid (`on_ref_grid`) and tracked with the anatomy's
  tissue fractions, so every method has the same stopping rule.
- **Tracker** (`run_tracking`). DIPY particle-filter tractography (Girard et al. 2014) with the continuous map
  criterion built from the wm/gm/csf fractions: step `step_mm` (0.5 mm), maximum turn `max_angle` (20°) per step,
  lengths `min_len_mm`–`max_len_mm` (3–120 mm), back-tracking `pft_back_mm` (2 mm), front-tracking `pft_front_mm`
  (1 mm), `particles` (15). Directions are drawn from the fODF on the 724 directions of `symmetric724`. When a path
  stops in white matter or CSF the particle filter backs up and retries. Only streamlines that end in grey matter
  are returned.
- **Probabilistic comparison.** A method name ending in `+prob` (for example `csd+prob`) tracks the same fODF with
  DIPY's plain probabilistic tracker, with the same seeds, step, angle and stopping, but without particle
  filtering: paths that stop in the wrong tissue are dropped. On macOS this tracker runs in a separate Python
  process (`python -m retinopath.track in.npz out.npz`), because DIPY's threaded tracker and PyTorch load two
  OpenMP runtimes that crash in one process; the streamlines are identical.
- **Endpoints** (`endpoints`). Each end of a streamline is assigned to the nearest white-surface vertex within
  `endpoint_tol_mm` (2.5 mm), or −1.
- **Area pairs** (`area_pairs`). A streamline is kept when both ends are assigned, lie in two different areas of
  V1, V2, V3, and are in the same hemisphere. Ends are ordered so that `a` is in the lower area; `pair` is 12, 23
  or 13.
- **Density map** (`density_map`). Streamlines per voxel of the crop grid, each streamline counted once per voxel.
- **Stored** (`tract_<protocol>_<method>.npz`): `a`, `b`, `pair` and `length` per kept streamline, the counts of
  seeds, of streamlines returned and of streamlines with both ends on the white surface, the density map, and the
  kept streamlines themselves.
- **Proximity bound** (`proximity_tract`, method `proximity`). Knows nothing about diffusion: each seed vertex is
  joined to a vertex of another area in the same hemisphere within 30 mm, chosen with probability
  ∝ exp(−distance / 5 mm). Its score shows how much nearby cortex gives for free.

## Retinotopic keys (`metrics.get_key`)

For every vertex: polar angle and eccentricity from a key source, the visual-field position
x = ecc · cos(angle), y = ecc · sin(angle) in degrees, and a validity mask: in V1–V3, finite angle, variance
explained ≥ `prf.r2_min` (10%), and `prf.ecc_min` ≤ eccentricity ≤ `prf.ecc_max` (0.5–7°; 4.2° for NSD, whose
stimulus was smaller).

| Key | Source | Columns |
|---|---|---|
| Measured | The subject's own pRF fit, all runs | `rsi_*`, `ree_*`, … |
| Group average | HCP group-average pRF maps (ID 999999) on the subject's vertices (`pipeline.group_key_source`) | `template_rsi_ree_*` |
| Atlas | Benson et al. 2014 atlas ([05-preprocessing.md](05-preprocessing.md#atlas-key-atlaspy)) | `atlas_rsi_ree_*` |

For HCP subjects the group average includes the subject; for NSD it is an independent group. The split-half fits
(`fit=1`, `fit=2`) are used for the noise ceiling and the robustness analysis.

## Endpoint error, null and RSI

Only streamlines with both ends valid in the key are scored.

- **Endpoint error** (`endpoint_error`), per streamline: the visual-field distance between the pRF centres of its
  two ends (`ree`, degrees), the absolute log ratio of their eccentricities (`dlogecc`), and the polar-angle
  difference (`dang`, degrees, wrapped to 0–180).
- **Distance-matched null** (`null_error`). Neighbouring cortex has similar retinotopy, so a short streamline has a
  small error for free. For each streamline, end `a` stays and end `b` is replaced by `n_null` (100) random valid
  vertices of the same area and hemisphere as `b` whose distance from `a` matches the real distance d within
  max(`null_tol` × d, `null_tol_min_mm`) (10%, at least 1 mm). The matches are drawn uniformly without
  replacement. Distance is straight-line, or for the geodesic version along the white surface (Dijkstra on the
  mesh graph of edge lengths, `build_graph`). A streamline with fewer than `n_null` matches gets NaN.
- **RSI** (`rsi`): `1 − median(observed error) / median(null error)` over the streamlines that have a full null;
  NaN with fewer than 20. 0 means the tractogram is no more retinotopic than distance alone predicts; 1 means
  every streamline joins points with the same visual-field position.

`retinotopic_scores` returns, for each area pair (`V1-V2`, `V2-V3`, `V1-V3`) and for `all`: the number of
streamlines; the median observed error, the median null error and the RSI for each of `ree`, `dlogecc`, `dang`;
the geodesic-null RSI of `ree`; and the coverage of the two areas (fraction of their valid vertices reached by at
least one streamline end). It also returns the binned AUC below.

## Other measures

- **Binned AUC** (`binned_auc`). Eccentricity is cut into `prf.n_ecc_bins` (6) bins at quantiles of the valid V1
  eccentricities. For each pair of areas the streamline counts form a 6 × 6 matrix (bin of end `a` × bin of end
  `b`); the score is the Mann–Whitney AUC separating "expected" cells (bins at most one apart) from "unexpected"
  ones (three or more apart). NaN with fewer than 20 streamlines.
- **Noise ceiling** (`noise_ceiling`). Median visual-field distance between the two split-half pRF estimates of
  the same vertex. Each half is noisier than the full fit, so it is an upper bound on what measurement noise adds
  to the endpoint error. HCP only.
- **Agreement with the reference tractogram** (HCP 3T sessions, every method except the reference itself and
  `proximity`), following the scilpy definitions: density-weighted Dice (`weighted_dice`; each density map is
  scaled to sum to 1, and the result is the mean of the two maps' mass inside the overlap) and bundle adjacency
  (`bundle_adjacency`; mean distance in mm from the voxels of one bundle outside the other bundle to the other
  bundle, averaged over both directions).
- **Voxel-wise accuracy** (`voxel_metrics`), in voxels with wm > 0.5 where both fODFs are non-empty: the mean
  angular correlation coefficient over SH orders above 0 (`acc`), and the median angle between the main peaks on
  `symmetric724` (`peak_err_deg`).

## Reproducibility (`pipeline.retest_scores`)

Between two scans of one person (HCP test vs retest; NSD run 1 vs run 2): weighted Dice and bundle adjacency of
the two V1–V3 tractograms, and the correlation of per-vertex streamline-end counts over V1–V3 (`vertex_count_r`).
HCP retest has its own anatomy, so the retest streamlines are first aligned rigidly to the test scan: a Kabsch fit
on the V1–V3 vertices, which correspond one-to-one on the fsLR mesh, with the determinant fixed to exclude a
reflection. `align_rms_mm` is the residual vertex distance after the fit. The two NSD runs already share one space.

## One result row (`pipeline.scores`)

`scores(cfg, ses, protocol, method)` tracks if needed and returns one row per session × protocol × method:
identifiers, tractogram counts, the measured-key scores, the noise ceiling, the template and atlas RSI with their
number of valid streamlines, and the comparison with the reference. The row is cached as
`scores_<protocol>_<method>_<hash>.json`. The hash covers the settings that change scores but not tractograms:
`prf`, `n_null`, `seed`, the reference, and which keys and nulls were used. Changing them recomputes scores from
the cached tractograms. Column names are listed in [08-code.md](08-code.md#result-columns).

## Statistics (`stats.py`, notebook 09)

The subject is the unit of analysis. The two training seeds of a network are averaged within subject before any
test.

- `paired_test(x, y, alternative, shift)`: Wilcoxon signed-rank test of the paired differences x − y (needs at
  least 6 subjects and a non-zero difference, otherwise p is NaN), with the median difference, the
  Hodges–Lehmann pseudo-median (median of all pairwise averages), the rank-biserial correlation, and a percentile
  bootstrap interval of the median (`stats.n_boot` resamples, 1,000 in dev). `shift` is added to the
  differences, which turns the test into a non-inferiority test.
- `correct`: Holm for the confirmatory family, Benjamini–Hochberg FDR for the exploratory grid.
- `mixed_model`: linear mixed model `rsi ~ C(method, Treatment('csd')) + C(protocol)` with a random intercept per
  subject, method effects relative to classical CSD.

### Confirmatory family (`confirmatory_tests.csv`, Holm, `significant` at `stats.alpha` 0.05)

- **H1.** For each area pair and for eccentricity (`rsi_dlogecc`) and polar angle (`rsi_dang`): reference RSI
  > 0, one-sided. Six tests.
- **H2.** For the `voxel` network on each clinical-grade protocol, relative to classical CSD on the same protocol:

      recovery of ACC = (ACC_net − ACC_csd) / (1 − ACC_csd)            (the reference has ACC 1)
      recovery of RSI = (RSI_net − RSI_csd) / (RSI_ref − RSI_csd)

  Subject × protocol cells where CSD loses less than 0.02 RSI against the reference have nothing to recover and
  are left out; the number left out is printed. Per subject the median over protocols is taken. Test: recovery of
  ACC > recovery of RSI, one-sided.
- **H3.** For each clinical-grade protocol: RSI of `path` > RSI of `voxel` (one-sided), and ACC of `path` not
  worse than ACC of `voxel` by more than `stats.acc_margin` (0.02): one-sided test of ACC_path − ACC_voxel + 0.02
  > 0.

### Other tests

- **Pairwise comparisons** (`pairwise_tests.csv`, FDR). RSI of every pair of methods on each clinical-grade protocol,
  two-sided, networks averaged over seeds; FDR over all pairs of all protocols. Drawn as supplementary figure S4.
- **Exploratory grid** (`exploratory_tests.csv`, FDR). Every method on every protocol against classical CSD on the
  same protocol, two-sided; networks are one method per variant (seeds averaged).
- **Mixed model** (`mixed_model.csv`). On the clinical-grade protocols and the methods that were run on all of
  them, so the design is balanced.
- **Other keys** (`alternative_key_tests.csv`, descriptive). `path` vs `voxel` with the group-average and with the
  atlas key, per clinical protocol; and the Spearman correlation between measured-key and atlas-key RSI over all
  rows of the reference, protocol and network tables.
- **External data** (`external_tests.csv`, descriptive). On HCP 7T and NSD: `path` − `voxel` and `path` − CSD,
  paired. With 8 NSD subjects these are reported with their intervals and are not part of the confirmatory family.
