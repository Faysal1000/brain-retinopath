# Tractography, scoring and statistics

How each fODF becomes a tractogram, how a tractogram is scored against retinotopy, and how the scores are tested.
Code: [track.py](../src/retinopath/track.py), [metrics.py](../src/retinopath/metrics.py),
[stats.py](../src/retinopath/stats.py), `tract`, `scores` and `retest_scores` in
[pipeline.py](../src/retinopath/pipeline.py), `path_error` in [model.py](../src/retinopath/model.py) and
`path_errors` in [train.py](../src/retinopath/train.py), and notebook 09.

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
  (1 mm), `particles` (15). Directions are drawn from the fODF on the 724 directions of `symmetric724`, inside the
  turning cone, with probability proportional to amplitude; directions below `pmf_threshold` (0.7) of the largest
  amplitude in the voxel are never drawn. DIPY's default of 0.1 keeps almost every direction of the cone, because
  an order-8 fODF lobe is wider than the cone, and the streamlines then scatter; the value was chosen on the
  development subjects ([09-development.md](09-development.md)). When a path stops in white matter or CSF the
  particle filter backs up and retries. Only streamlines that end in grey matter are returned.
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
  seeds, of streamlines returned and of streamlines with both ends on the white surface, the density map, the
  kept streamlines themselves, and the tracking settings. A file made with other `tracking` settings or another
  `seeds_per_vertex` is tracked again when it is next asked for.
- **Proximity bound** (`proximity_tract`, method `proximity`). Knows nothing about diffusion: each seed vertex is
  joined to a vertex of another area in the same hemisphere within 30 mm, chosen with probability
  ∝ exp(−distance / 5 mm). Its score shows how much nearby cortex gives for free.

## Retinotopic keys (`metrics.get_key`)

For every vertex: polar angle and eccentricity from a key source, the visual-field position
x = ecc · cos(angle), y = ecc · sin(angle) in degrees, and a validity mask: in V1–V3, finite angle, variance
explained ≥ `prf.r2_min` (10%), and `prf.ecc_min` ≤ eccentricity ≤ `prf.ecc_max` (0.5–7°; 4.2° for NSD, whose
stimulus was smaller). `prf.meridian_margin_deg` (0 in the main analysis) also drops vertices whose polar angle is
within that many degrees of the horizontal or vertical meridian, the borders between the areas, where a vertex
is most easily given to the wrong area.

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
  small error for free. For each streamline, end `a` stays and end `b` is replaced by `n_null` (20) random valid
  vertices of the same area and hemisphere as `b` whose distance from `a` matches the real distance d within
  max(`null_tol` × d, `null_tol_min_mm`) (10%, at least 1 mm). The matches are drawn uniformly without
  replacement. Distance is straight-line, or for the geodesic version along the white surface (Dijkstra on the
  mesh graph of edge lengths, `build_graph`; `Geodesic` keeps the distances from each vertex, so all tractograms
  of a session share them). A streamline with fewer than `n_null` matches gets NaN. The tolerance band holds few
  mesh vertices, so `n_null` stays small: with 20 about 40% of streamlines have a full null (most of the V1–V3
  ones, fewer of the short V1–V2 and V2–V3 ones); with 100 almost none do.
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
- **Path error** (`model.path_error`, `train.path_errors`, notebook 06), for classical CSD and every network on
  the clinical-grade protocols of the test subjects. One seed per V1–V3 white-surface vertex,
  `tracking.seed_depth_mm` below the surface along the normal. From each seed the soft tracker of the
  path-consistency loss ([06-network.md](06-network.md#soft-tracker-pathtracer)) follows the reference fODF and the
  method's fODF (on the crop grid) from the same start direction, taken from the reference fODF, with the settings
  of the training loss: `model.path.cone_deg`, `sharpen`, and `steps_end` (40) steps of `step_mm` (0.625 mm). A
  point counts while the reference path is more than one voxel inside the crop and in white matter (wm > 0.1);
  all later points are dropped, and so is the seed itself. `path_err_mm` is the mean distance (mm) between the two
  paths over all counted points; `path_end_err_mm` is the mean over paths of the distance at the last counted
  point; `n_paths` is the number of paths with at least one counted point. The distance is Euclidean, not the
  Huber distance of the loss. Because the tracer and path length are those of the training loss, the path error
  measures how well that objective generalises to test subjects; RSI is independent of training. Cached as
  `patherr_<protocol>_<method>.json` beside the fODF and computed again when those settings change; the table is
  `path_error.csv`.

## Reproducibility (`pipeline.retest_scores`)

Between two scans of one person (HCP test vs retest; NSD run 1 vs run 2): weighted Dice and bundle adjacency of
the two V1–V3 tractograms, and the correlation of per-vertex streamline-end counts over V1–V3 (`vertex_count_r`).
HCP retest has its own anatomy, so the retest streamlines are first aligned rigidly to the test scan: a Kabsch fit
on the V1–V3 vertices, which correspond one-to-one on the fsLR mesh, with the determinant fixed to exclude a
reflection. `align_rms_mm` is the residual vertex distance after the fit. The two NSD runs already share one space.

## Robustness of the reference RSI (`pipeline.robustness_rows`, notebook 08)

The reference tractogram of every test subject is scored again with one choice changed at a time
(`robustness.csv`, one row per subject and analysis):

- pRF quality: variance explained ≥ 5% and ≥ 20%.
- Null tolerance: 5% and 20% of the endpoint distance.
- Eccentricity range: ≤ 5°, ≥ 1°, and 1–6° cut into 10 bins (changes the binned AUC).
- Meridians: vertices within 15° of polar angle of the horizontal or vertical meridian left out
  (`prf.meridian_margin_deg`), so that streamline ends near an area border, which may be assigned to the wrong
  area, do not carry the result.
- Sampling threshold: the reference fODF is tracked again with DIPY's default `pmf_threshold` of 0.1.
- Tracking step: the reference fODF is tracked again from the same seeds with a 0.2 mm step. The turning limit is
  lowered so that the tightest allowed curve keeps the radius of the main tracker, step / (2 sin(max_angle / 2)):
  8° per step for the default 0.5 mm and 20°.
- Key from each half of the pRF runs.
- Dorsal, ventral, left and right streamlines separately (subsets with more than 50 streamlines).

## Sampling threshold and lobe width (`pipeline.sampling_rows`, notebook 08)

The sampling threshold of the tracker was chosen on the development subjects, so the comparison of the networks
is repeated with DIPY's default. For classical CSD and every training seed of `path` and `voxel` on the
clinical-grade protocols, the same fODF is tracked again from the same seeds with `pmf_threshold` 0.1 and scored
(`sampling.csv`, one row per subject, protocol and method; the tractograms are cached beside those of the main
tracker with the suffix `_pmf_threshold0.1`).

The same table holds `lobe_width` (`metrics.lobe_width`): the fraction of the 724 sphere directions with an
amplitude of at least `tracking.pmf_threshold` (0.7) of the largest amplitude in the voxel, averaged over the
voxels used for ACC. These are the directions the main tracker can draw from before the turning cone is applied.
A narrower lobe alone raises RSI with a sampling tracker ([09-development.md](09-development.md)), so the width
shows whether a difference in RSI between two methods can come from how peaked their lobes are and not from
where they point.

## One result row (`pipeline.scores`)

`scores(cfg, ses, protocol, method)` tracks if needed and returns one row per session × protocol × method:
identifiers, tractogram counts, the measured-key scores, the noise ceiling, the template and atlas RSI with their
number of valid streamlines, and the comparison with the reference. The row is cached as
`scores_<protocol>_<method>_<hash>.json`. The hash covers the tracking settings and the settings that change
scores but not tractograms: `prf`, `n_null`, `seed`, the reference, and which keys and nulls were used. Changing
the latter recomputes scores from the cached tractograms. Column names are listed in [08-code.md](08-code.md#result-columns).

## Statistics (`stats.py`, notebook 09)

The subject is the unit of analysis. The training seeds of a network (`train_seeds`) are averaged within subject
before any test.

- `paired_test(x, y, alternative, shift)`: Wilcoxon signed-rank test of the paired differences x − y (needs at
  least 6 subjects and a non-zero difference, otherwise p is NaN), with the median difference, the
  Hodges–Lehmann pseudo-median (median of all pairwise averages), the rank-biserial correlation, and a percentile
  bootstrap interval of the median (`stats.n_boot` resamples, 1,000 in dev). `shift` is added to the
  differences for the test only, which turns it into a non-inferiority test; the median, the pseudo-median and
  the interval describe the differences themselves.
- `correct`: Holm for the confirmatory family, Benjamini–Hochberg FDR for the exploratory grid.
- `mixed_model`: linear mixed model `rsi ~ C(method, Treatment('csd')) + C(protocol)` with a random intercept per
  subject, method effects relative to classical CSD.

### Confirmatory family (`confirmatory_tests.csv`, Holm, `significant` at `stats.alpha` 0.05)

Six H1 tests, one H2 test and two H3 tests per clinical-grade protocol (13 with P1, P2, P3), Holm-corrected
together.

- **H1.** For each area pair and for eccentricity (`rsi_dlogecc`) and polar angle (`rsi_dang`): reference RSI
  > 0, one-sided. Six tests.
- **H2.** For the `voxel` network on each clinical-grade protocol, relative to classical CSD on the same protocol:

      recovery of ACC = (ACC_net − ACC_csd) / (1 − ACC_csd)            (the reference has ACC 1)
      recovery of RSI = (RSI_net − RSI_csd) / (RSI_ref − RSI_csd)

  Subject × protocol cells where CSD loses less than 0.02 RSI against the reference have nothing to recover and
  are left out of both recoveries, so ACC and RSI use the same cells; the number left out is printed. Per subject
  the median over the remaining protocols is taken. Test: recovery of ACC > recovery of RSI, one-sided.
- **H3.** For each clinical-grade protocol: RSI of `path` > RSI of `voxel` (one-sided), and ACC of `path` not
  worse than ACC of `voxel` by more than `stats.acc_margin` (0.02): one-sided test of ACC_path − ACC_voxel + 0.02
  > 0.

In H2 and H3 a network's score for a subject is the mean over its training seeds, and a subject enters a test
only when every seed has a score. The number of tests and of tests without a p-value is printed.

### Comparison with classical CSD (`csd_comparison_tests.csv`, Holm within this family)

Five one-sided tests on each clinical-grade protocol. A network's score for a subject is the mean over its training
seeds, and a subject enters a test only when every seed has a score.

- RSI of `voxel` < RSI of classical CSD: training on voxel error alone lowers retinotopic specificity.
- RSI of `path` not worse than RSI of classical CSD by more than `stats.rsi_margin` (0.05): test of
  RSI_path − RSI_csd + 0.05 > 0. The margin is about a tenth of the reference RSI and less than the loss seen with
  the `voxel` network on the development subjects ([09-development.md](09-development.md)).
- ACC of `path` > ACC of classical CSD.
- Path error (`path_err_mm`) of `path` < path error of classical CSD.
- Path error of `path` < path error of `voxel`.

The first test asks whether voxel-only training lowers RSI; the second and third, whether the path loss avoids
that while keeping ACC above classical CSD; the last two, whether the path network follows the reference paths
more closely than the fODF it starts from and than the same network trained on voxel error alone.

### Training size (`training_size_tests.csv`)

Both families above are tested on the networks trained with 12 subjects (`n_train` in `config.yaml`). The whole
experiment is repeated with 35 and 100 training subjects, on the same test subjects
([04-configuration.md](04-configuration.md#training-size)). Notebook 09 stacks the H3 tests and the comparison
with classical CSD of every size into one table, with the column `n_train` and the Holm correction made within
each size and family. These rows describe how the effects change with the amount of training data; they are not
part of the confirmatory family. Drawn as Figure 8.

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
- **Sampling threshold and lobe width** (`sampling_tests.csv`, descriptive). From `sampling.csv`, per clinical
  protocol, networks averaged over seeds: RSI at the sampling threshold of 0.1 and lobe width, each as `path` −
  `voxel`, `path` − CSD and `voxel` − CSD, paired and two-sided.
- **External data** (`external_tests.csv`, descriptive). On HCP 7T and NSD: `path` − `voxel` and `path` − CSD,
  paired. With 8 NSD subjects these are reported with their intervals and are not part of the confirmatory family.
