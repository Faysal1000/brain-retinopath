"""Retinotopic fidelity of streamlines against measured pRF maps, distance-matched nulls, reproducibility, voxel metrics."""

import numpy as np
from scipy import ndimage, sparse, stats
from scipy.sparse.csgraph import dijkstra
from .recon import sh_matrix, tracking_sphere

PAIRS = {12: "V1-V2", 23: "V2-V3", 13: "V1-V3"}


# ---------------------------------------------------------------- retinotopic key
def get_key(anat, cfg, fit=0, group=None):
    """Visual-field position of every vertex. Returns x = ecc cos(ang), y = ecc sin(ang) (deg), ecc, ang (deg), and a
    valid mask (V1-V3, finite angle, r2 >= r2_min, ecc_min <= ecc <= ecc_max, at least meridian_margin_deg of polar
    angle away from the horizontal and vertical meridians, where the areas border each other).
    fit: 0 = all runs, 1/2 = split halves. group: optional anat-like dict holding group-average maps (template key)."""
    source = group if group is not None else anat
    ang, ecc, r2 = source["prf_ang"][:, fit], source["prf_ecc"][:, fit], source["prf_r2"][:, fit]
    limits = cfg.prf
    to_meridian = np.abs((ang + 45) % 90 - 45)  # polar-angle distance to the nearest meridian (deg)
    valid = (
        (anat["area"] > 0)
        & np.isfinite(ang)
        & (r2 >= limits.r2_min)
        & (ecc >= limits.ecc_min)
        & (ecc <= limits.ecc_max)
        & (to_meridian >= limits.meridian_margin_deg)
    )
    return dict(x=ecc * np.cos(np.radians(ang)), y=ecc * np.sin(np.radians(ang)), ecc=ecc, ang=ang, valid=valid)


def endpoint_error(key, lower, higher):
    """Per-streamline error between the two ends: visual-field distance (deg), |log eccentricity ratio|, polar-angle
    difference (deg)."""
    field_distance = np.hypot(key["x"][lower] - key["x"][higher], key["y"][lower] - key["y"][higher])
    log_ecc_ratio = np.abs(np.log(key["ecc"][lower] / key["ecc"][higher]))
    angle_difference = np.abs((key["ang"][lower] - key["ang"][higher] + 180) % 360 - 180)
    return field_distance, log_ecc_ratio, angle_difference


# ---------------------------------------------------------------- distance-matched null
def build_graph(anat):
    """Edge-length graph over the white-surface mesh (hemispheres are separate components), for surface distances."""
    faces = anat["faces"]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    lengths = np.linalg.norm(anat["verts"][edges[:, 0]] - anat["verts"][edges[:, 1]], axis=1)
    graph = sparse.coo_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(len(anat["verts"]),) * 2).tocsr()
    return graph.maximum(graph.T)


class Geodesic:
    """Surface distances between V1-V3 vertices of one session. The distances from a vertex are computed the first
    time it is asked for and kept, so every tractogram of the session after the first reuses them."""

    def __init__(self, anat):
        self.anat, self.graph, self.rows = anat, None, {}
        self.roi = np.flatnonzero(anat["area"] > 0)
        self.pos = np.full(len(anat["verts"]), -1)
        self.pos[self.roi] = np.arange(len(self.roi))

    def table(self, sources, candidates):
        """(len(src), len(cand)) distances in mm; src and cand are V1-V3 vertex indices."""
        new = [int(vertex) for vertex in np.unique(sources) if int(vertex) not in self.rows]
        if new:
            if self.graph is None:
                self.graph = build_graph(self.anat)
            for vertex, row in zip(new, dijkstra(self.graph, directed=False, indices=new)):
                self.rows[vertex] = row[self.roi]
        return np.stack([self.rows[int(vertex)] for vertex in sources])[:, self.pos[candidates]]


def null_error(anat, key, lower, higher, cfg, rng, graph=None, chunk=2000):
    """Surrogate errors: the higher-area end moves to n_null random valid vertices of its area and hemisphere with
    |d - d_obs| <= max(null_tol * d_obs, null_tol_min_mm) from the lower end (straight-line, or surface distance
    with a Geodesic `graph`), drawn without replacement. Returns (ree, dlog, dang), each (n, n_null); NaN rows
    where there are fewer matches or the end is not valid."""
    from scipy.spatial.distance import cdist

    count, n_null = len(lower), cfg.n_null
    out = [np.full((count, n_null), np.nan, np.float32) for _ in range(3)]
    endpoint_group = (
        anat["hemi"][higher].astype(int) * 10 + anat["area"][higher]
    )  # hemi * 10 + area of the higher endpoint
    for code in np.unique(endpoint_group):
        candidates = np.flatnonzero(key["valid"] & (anat["hemi"] == code // 10) & (anat["area"] == code % 10))
        group_rows = np.flatnonzero(
            (endpoint_group == code) & key["valid"][higher]
        )  # the higher endpoint itself must be a valid vertex
        if len(candidates) < n_null:
            continue
        position = np.full(len(anat["verts"]), -1)
        position[candidates] = np.arange(len(candidates))
        step = (
            chunk if graph is None else max(50, int(2e8 // (8 * len(anat["verts"]))))
        )  # geodesic table (rows x n_verts float64) about 200 MB
        for start in range(0, len(group_rows), step):
            rows = group_rows[start : start + step]
            if graph is None:
                distances = cdist(anat["verts"][lower[rows]], anat["verts"][candidates])
            else:
                unique_lower, inverse = np.unique(lower[rows], return_inverse=True)
                distances = graph.table(unique_lower, candidates)[inverse]
            observed = distances[np.arange(len(rows)), position[higher[rows]]]
            tolerance = np.maximum(cfg.prf.null_tol * observed, cfg.prf.null_tol_min_mm)
            score = np.where(
                np.abs(distances - observed[:, None]) <= tolerance[:, None], rng.random(distances.shape), np.inf
            )
            picked = np.argpartition(score, n_null - 1, axis=1)[:, :n_null]
            enough = np.isfinite(np.take_along_axis(score, picked, 1)).all(1)
            repeated_lower = np.repeat(lower[rows][:, None], n_null, 1)
            for target, error in zip(out, endpoint_error(key, repeated_lower, candidates[picked])):
                error = error.astype(np.float32)
                error[~enough] = np.nan
                target[rows] = error
    return out


def rsi(observed, null):
    """Retinotopic specificity index: 1 - median(observed) / median(null). 0 = no better than proximity."""
    valid = np.isfinite(observed) & np.isfinite(null).all(1)
    if valid.sum() < 20:
        return np.nan
    return float(1 - np.median(observed[valid]) / np.median(null[valid]))


def retinotopic_scores(anat, lower, higher, pair, cfg, key=None, graph=None, seed=0):
    """All per-subject functional scores for one tractogram (end vertices `lower`, `higher`; pair codes 12/23/13). Only streamlines
    with both ends valid in `key` are scored. graph: None, True (made here) or a Geodesic of this session; adds the
    geodesic-null RSI. Returns a flat dict."""
    key = key or get_key(anat, cfg)
    rng = np.random.default_rng(seed)
    both_valid = key["valid"][lower] & key["valid"][higher]
    lower, higher, pair = lower[both_valid], higher[both_valid], pair[both_valid]
    observed = endpoint_error(key, lower, higher)
    null = null_error(anat, key, lower, higher, cfg, rng)
    if graph is True:
        graph = Geodesic(anat)
    null_geodesic = null_error(anat, key, lower, higher, cfg, rng, graph) if graph is not None else None
    out = dict(n_valid=int(len(lower)))
    for code, name in list(PAIRS.items()) + [(0, "all")]:
        selected = np.ones(len(lower), bool) if code == 0 else pair == code
        out[f"n_{name}"] = int(selected.sum())
        for quantity, observed_errors, null_errors in zip(("ree", "dlogecc", "dang"), observed, null):
            out[f"{quantity}_median_{name}"] = float(np.median(observed_errors[selected])) if selected.any() else np.nan
            out[f"{quantity}_null_median_{name}"] = (
                float(np.nanmedian(null_errors[selected])) if selected.any() else np.nan
            )
            out[f"rsi_{quantity}_{name}"] = rsi(observed_errors[selected], null_errors[selected])
        if null_geodesic is not None:
            out[f"rsi_ree_geodesic_{name}"] = rsi(observed[0][selected], null_geodesic[0][selected])
        if code:  # coverage of the two areas of this pair
            for area, ends in ((code // 10, lower[selected]), (code % 10, higher[selected])):
                area_valid = key["valid"] & (anat["area"] == area)
                out[f"coverage_{name}_V{area}"] = float(len(np.unique(ends)) / max(area_valid.sum(), 1))
    out.update(binned_auc(anat, key, lower, higher, pair, cfg))
    return out


def noise_ceiling(anat, cfg):
    """Median visual-field distance between the two split-half pRF estimates of the same vertex (deg). Each half
    is noisier than the full fit, so this is an upper bound on what measurement noise adds to the endpoint error."""
    first_half, second_half = get_key(anat, cfg, 1), get_key(anat, cfg, 2)
    valid = first_half["valid"] & second_half["valid"]
    return (
        float(
            np.median(
                np.hypot(
                    first_half["x"][valid] - second_half["x"][valid], first_half["y"][valid] - second_half["y"][valid]
                )
            )
        )
        if valid.any()
        else np.nan
    )


def binned_auc(anat, key, lower, higher, pair, cfg):
    """Binned score: eccentricity-by-eccentricity streamline counts between two areas; AUC separating
    'expected' entries (bins at most one apart) from 'unexpected' ones (three or more apart). Bin edges are
    quantiles of valid V1 eccentricities (n_ecc_bins bins). NaN for pairs with fewer than 20 streamlines."""
    n_bins = cfg.prf.n_ecc_bins
    out = {}
    edges = (
        np.quantile(key["ecc"][key["valid"] & (anat["area"] == 1)], np.linspace(0, 1, n_bins + 1))
        if key["valid"].any()
        else None
    )
    for code, name in PAIRS.items():
        selected = pair == code
        if edges is None or selected.sum() < 20:
            out[f"auc_binned_{name}"] = np.nan
            continue
        lower_bin = np.clip(np.digitize(key["ecc"][lower[selected]], edges[1:-1]), 0, n_bins - 1)
        higher_bin = np.clip(np.digitize(key["ecc"][higher[selected]], edges[1:-1]), 0, n_bins - 1)
        counts = np.zeros((n_bins, n_bins))
        np.add.at(counts, (lower_bin, higher_bin), 1)
        row, column = np.indices(counts.shape)
        expected, unexpected = counts[np.abs(row - column) <= 1], counts[np.abs(row - column) >= 3]
        out[f"auc_binned_{name}"] = float(
            stats.mannwhitneyu(expected, unexpected).statistic / (len(expected) * len(unexpected))
        )
    return out


# ---------------------------------------------------------------- reproducibility (definitions follow scilpy)
def weighted_dice(first, second):
    """Density-weighted overlap of two bundles; each density map is first scaled to sum to 1."""
    if first.sum() == 0 or second.sum() == 0:
        return np.nan
    first, second = first / first.sum(), second / second.sum()
    overlap = (first > 0) & (second > 0)
    return float((first[overlap].sum() + second[overlap].sum()) / 2)


def bundle_adjacency(first, second, voxel_mm):
    """Mean distance (mm) from the voxels of one bundle that are outside the other to the other bundle, averaged
    over both directions. 0 = identical masks."""
    first_mask, second_mask = first > 0, second > 0
    if not first_mask.any() or not second_mask.any():
        return np.nan
    distance_to_first = ndimage.distance_transform_edt(~first_mask) * voxel_mm
    distance_to_second = ndimage.distance_transform_edt(~second_mask) * voxel_mm
    first_outside = distance_to_second[first_mask & ~second_mask]
    second_outside = distance_to_first[second_mask & ~first_mask]
    return float(
        (
            (first_outside.mean() if len(first_outside) else 0.0)
            + (second_outside.mean() if len(second_outside) else 0.0)
        )
        / 2
    )


# ---------------------------------------------------------------- voxel-wise accuracy
def voxel_metrics(pred, ref, mask):
    """pred, ref: (n_coef, X, Y, Z) SH on the same grid. Returns mean angular correlation coefficient (SH l > 0) and
    median angle between main peaks (deg, on symmetric724) inside `mask`. Voxels where either fODF is empty
    (outside its fitting mask) are left out."""
    mask = mask & (np.abs(pred[1:]).sum(0) > 0) & (np.abs(ref[1:]).sum(0) > 0)
    predicted, reference = pred[1:, mask].T.astype(np.float64), ref[1:, mask].T.astype(np.float64)
    acc = (predicted * reference).sum(1) / (
        np.linalg.norm(predicted, axis=1) * np.linalg.norm(reference, axis=1) + 1e-12
    )
    directions = tracking_sphere().vertices
    sh_basis = sh_matrix(directions, int((np.sqrt(8 * pred.shape[0] + 1) - 3) / 2))
    predicted_peak = directions[(pred[:, mask].T @ sh_basis.T).argmax(1)]
    reference_peak = directions[(ref[:, mask].T @ sh_basis.T).argmax(1)]
    angle = np.degrees(np.arccos(np.clip(np.abs((predicted_peak * reference_peak).sum(1)), 0, 1)))
    return float(np.mean(acc)), float(np.median(angle))


def lobe_width(sh, mask, threshold):
    """Fraction of sphere directions (symmetric724) with amplitude of at least `threshold` x the largest amplitude of
    the voxel, averaged over `mask`: how many directions a tracker with this sampling threshold can draw from.
    sh: (n_coef, X, Y, Z)."""
    mask = mask & (np.abs(sh[1:]).sum(0) > 0)
    amplitude = sh[:, mask].T @ sh_matrix(tracking_sphere().vertices, int((np.sqrt(8 * sh.shape[0] + 1) - 3) / 2)).T
    peak = amplitude.max(1, keepdims=True)
    return float((amplitude[peak[:, 0] > 0] >= threshold * peak[peak[:, 0] > 0]).mean())
