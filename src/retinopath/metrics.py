"""Retinotopic fidelity of streamlines against measured pRF maps, distance-matched nulls, reproducibility, voxel metrics."""
import numpy as np
from scipy import ndimage, sparse, stats
from scipy.sparse.csgraph import dijkstra
from .recon import sh_matrix, tracking_sphere

PAIRS = {12: "V1-V2", 23: "V2-V3", 13: "V1-V3"}


# ---------------------------------------------------------------- retinotopic key
def get_key(anat, cfg, fit=0, group=None):
    """Visual-field position of every vertex. Returns x = ecc cos(ang), y = ecc sin(ang) (deg), ecc, ang (deg), and a
    valid mask (V1-V3, finite angle, r2 >= r2_min, ecc_min <= ecc <= ecc_max).
    fit: 0 = all runs, 1/2 = split halves. group: optional anat-like dict holding group-average maps (template key)."""
    src = group if group is not None else anat
    ang, ecc, r2 = src["prf_ang"][:, fit], src["prf_ecc"][:, fit], src["prf_r2"][:, fit]
    p = cfg.prf
    valid = (anat["area"] > 0) & np.isfinite(ang) & (r2 >= p.r2_min) & (ecc >= p.ecc_min) & (ecc <= p.ecc_max)
    return dict(x=ecc * np.cos(np.radians(ang)), y=ecc * np.sin(np.radians(ang)), ecc=ecc, ang=ang, valid=valid)


def endpoint_error(key, a, b):
    """Per-streamline error between the two ends: visual-field distance (deg), |log eccentricity ratio|, polar-angle
    difference (deg)."""
    ree = np.hypot(key["x"][a] - key["x"][b], key["y"][a] - key["y"][b])
    dlog = np.abs(np.log(key["ecc"][a] / key["ecc"][b]))
    dang = np.abs((key["ang"][a] - key["ang"][b] + 180) % 360 - 180)
    return ree, dlog, dang


# ---------------------------------------------------------------- distance-matched null
def build_graph(anat):
    """Edge-length graph over the white-surface mesh (hemispheres are separate components), for surface distances."""
    f = anat["faces"]; e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    w = np.linalg.norm(anat["verts"][e[:, 0]] - anat["verts"][e[:, 1]], axis=1)
    g = sparse.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(len(anat["verts"]),) * 2).tocsr()
    return g.maximum(g.T)


def null_error(anat, key, a, b, cfg, rng, graph=None, chunk=2000):
    """Surrogate errors: endpoint a stays, endpoint b moves to a random valid vertex of the same area and hemisphere
    lying at (nearly) the same distance from a. Distance is straight-line, or along the surface if `graph` is given.
    A candidate matches when |d - d_obs| <= max(null_tol * d_obs, null_tol_min_mm); n_null of the matches are drawn
    uniformly without replacement (random scores, smallest k kept).
    Returns three (n, n_null) arrays (ree, dlog, dang); rows with fewer than n_null matches, or whose b is not a
    valid vertex, are NaN."""
    from scipy.spatial.distance import cdist
    n, k = len(a), cfg.n_null
    out = [np.full((n, k), np.nan, np.float32) for _ in range(3)]
    group = anat["hemi"][b].astype(int) * 10 + anat["area"][b]            # hemi * 10 + area of endpoint b
    for g in np.unique(group):
        cand = np.flatnonzero(key["valid"] & (anat["hemi"] == g // 10) & (anat["area"] == g % 10))
        rows_g = np.flatnonzero((group == g) & key["valid"][b])               # b itself must be a valid vertex
        if len(cand) < k:
            continue
        pos = np.full(len(anat["verts"]), -1); pos[cand] = np.arange(len(cand))
        step = chunk if graph is None else max(50, int(2e8 // (8 * len(anat["verts"]))))      # geodesic table (rows x n_verts float64) about 200 MB
        for s in range(0, len(rows_g), step):
            rows = rows_g[s:s + step]
            if graph is None:
                D = cdist(anat["verts"][a[rows]], anat["verts"][cand])
            else:
                ua, inv = np.unique(a[rows], return_inverse=True)
                D = dijkstra(graph, directed=False, indices=ua)[:, cand][inv]
            d_obs = D[np.arange(len(rows)), pos[b[rows]]]
            tol = np.maximum(cfg.prf.null_tol * d_obs, cfg.prf.null_tol_min_mm)
            score = np.where(np.abs(D - d_obs[:, None]) <= tol[:, None], rng.random(D.shape), np.inf)
            pick = np.argpartition(score, k - 1, axis=1)[:, :k]
            enough = np.isfinite(np.take_along_axis(score, pick, 1)).all(1)
            aa = np.repeat(a[rows][:, None], k, 1)
            for o, e in zip(out, endpoint_error(key, aa, cand[pick])):
                e = e.astype(np.float32); e[~enough] = np.nan; o[rows] = e
    return out


def rsi(obs, null):
    """Retinotopic specificity index: 1 - median(observed) / median(null). 0 = no better than proximity."""
    ok = np.isfinite(obs) & np.isfinite(null).all(1)
    if ok.sum() < 20:
        return np.nan
    return float(1 - np.median(obs[ok]) / np.median(null[ok]))


def retinotopic_scores(anat, a, b, pair, cfg, key=None, graph=None, seed=0):
    """All per-subject functional scores for one tractogram (end vertices a, b; pair codes 12/23/13). Only streamlines
    with both ends valid in `key` are scored. graph: None, True (build it here) or a build_graph result; adds the
    geodesic-null RSI. Returns a flat dict."""
    key = key or get_key(anat, cfg)
    rng = np.random.default_rng(seed)
    both = key["valid"][a] & key["valid"][b]
    a, b, pair = a[both], b[both], pair[both]
    obs = endpoint_error(key, a, b)
    nul = null_error(anat, key, a, b, cfg, rng)
    if graph is True:
        graph = build_graph(anat)
    nul_g = null_error(anat, key, a, b, cfg, rng, graph) if graph is not None else None
    out = dict(n_valid=int(len(a)))
    for code, name in list(PAIRS.items()) + [(0, "all")]:
        m = np.ones(len(a), bool) if code == 0 else pair == code
        out[f"n_{name}"] = int(m.sum())
        for q, o, nu in zip(("ree", "dlogecc", "dang"), obs, nul):
            out[f"{q}_median_{name}"] = float(np.median(o[m])) if m.any() else np.nan
            out[f"{q}_null_median_{name}"] = float(np.nanmedian(nu[m])) if m.any() else np.nan
            out[f"rsi_{q}_{name}"] = rsi(o[m], nu[m])
        if nul_g is not None:
            out[f"rsi_ree_geodesic_{name}"] = rsi(obs[0][m], nul_g[0][m])
        if code:                                                        # coverage of the two areas of this pair
            for area, ends in ((code // 10, a[m]), (code % 10, b[m])):
                tot = key["valid"] & (anat["area"] == area)
                out[f"coverage_{name}_V{area}"] = float(len(np.unique(ends)) / max(tot.sum(), 1))
    out.update(binned_auc(anat, key, a, b, pair, cfg))
    return out


def noise_ceiling(anat, cfg):
    """Median visual-field distance between the two split-half pRF estimates of the same vertex (deg). Each half
    is noisier than the full fit, so this is an upper bound on what measurement noise adds to the endpoint error."""
    k1, k2 = get_key(anat, cfg, 1), get_key(anat, cfg, 2)
    ok = k1["valid"] & k2["valid"]
    return float(np.median(np.hypot(k1["x"][ok] - k2["x"][ok], k1["y"][ok] - k2["y"][ok]))) if ok.any() else np.nan


def binned_auc(anat, key, a, b, pair, cfg):
    """Binned score: eccentricity-by-eccentricity streamline counts between two areas; AUC separating
    'expected' entries (bins at most one apart) from 'unexpected' ones (three or more apart). Bin edges are
    quantiles of valid V1 eccentricities (n_ecc_bins bins). NaN for pairs with fewer than 20 streamlines."""
    nb = cfg.prf.n_ecc_bins; out = {}
    edges = np.quantile(key["ecc"][key["valid"] & (anat["area"] == 1)], np.linspace(0, 1, nb + 1)) if key["valid"].any() else None
    for code, name in PAIRS.items():
        m = pair == code
        if edges is None or m.sum() < 20:
            out[f"auc_binned_{name}"] = np.nan; continue
        ba = np.clip(np.digitize(key["ecc"][a[m]], edges[1:-1]), 0, nb - 1)
        bb = np.clip(np.digitize(key["ecc"][b[m]], edges[1:-1]), 0, nb - 1)
        M = np.zeros((nb, nb)); np.add.at(M, (ba, bb), 1)
        i, j = np.indices(M.shape); exp, unexp = M[np.abs(i - j) <= 1], M[np.abs(i - j) >= 3]
        out[f"auc_binned_{name}"] = float(stats.mannwhitneyu(exp, unexp).statistic / (len(exp) * len(unexp)))
    return out


# ---------------------------------------------------------------- reproducibility (definitions follow scilpy)
def weighted_dice(d1, d2):
    """Density-weighted overlap of two bundles; each density map is first scaled to sum to 1."""
    if d1.sum() == 0 or d2.sum() == 0:
        return np.nan
    d1, d2 = d1 / d1.sum(), d2 / d2.sum(); ov = (d1 > 0) & (d2 > 0)
    return float((d1[ov].sum() + d2[ov].sum()) / 2)


def bundle_adjacency(d1, d2, voxel_mm):
    """Mean distance (mm) from the voxels of one bundle that are outside the other to the other bundle, averaged
    over both directions. 0 = identical masks."""
    m1, m2 = d1 > 0, d2 > 0
    if not m1.any() or not m2.any():
        return np.nan
    t1 = ndimage.distance_transform_edt(~m1) * voxel_mm; t2 = ndimage.distance_transform_edt(~m2) * voxel_mm
    a = t2[m1 & ~m2]; b = t1[m2 & ~m1]
    return float(((a.mean() if len(a) else 0.0) + (b.mean() if len(b) else 0.0)) / 2)


# ---------------------------------------------------------------- voxel-wise accuracy
def voxel_metrics(pred, ref, mask):
    """pred, ref: (n_coef, X, Y, Z) SH on the same grid. Returns mean angular correlation coefficient (SH l > 0) and
    median angle between main peaks (deg, on symmetric724) inside `mask`. Voxels where either fODF is empty
    (outside its fitting mask) are left out."""
    mask = mask & (np.abs(pred[1:]).sum(0) > 0) & (np.abs(ref[1:]).sum(0) > 0)
    p, r = pred[1:, mask].T.astype(np.float64), ref[1:, mask].T.astype(np.float64)
    acc = (p * r).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(r, axis=1) + 1e-12)
    sph = tracking_sphere().vertices; B = sh_matrix(sph, int((np.sqrt(8 * pred.shape[0] + 1) - 3) / 2))
    pp = sph[(pred[:, mask].T @ B.T).argmax(1)]; pr = sph[(ref[:, mask].T @ B.T).argmax(1)]
    ang = np.degrees(np.arccos(np.clip(np.abs((pp * pr).sum(1)), 0, 1)))
    return float(np.mean(acc)), float(np.median(ang))
