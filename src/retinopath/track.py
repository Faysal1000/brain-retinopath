"""Tractography seeded under V1-V3 (particle filtering by default, plain probabilistic as a comparison) and mapping of
streamline ends to cortical vertices."""
import json, subprocess, sys, tempfile
import numpy as np
from dipy.tracking.stopping_criterion import CmcStoppingCriterion
from dipy.tracking.tracker import pft_tracking, probabilistic_tracking
from scipy.spatial import cKDTree
from .recon import tracking_sphere


def make_seeds(anat, cfg, seed=0):
    """Seeds just below the white surface of V1, V2 and V3 (world mm) and the vertex each one came from."""
    t = cfg.tracking; rng = np.random.default_rng(seed)
    v = np.flatnonzero(anat["area"] > 0).repeat(cfg.seeds_per_vertex)
    pts = anat["verts"][v] - t.seed_depth_mm * anat["normals"][v] + rng.normal(0, t.seed_jitter_mm, (len(v), 3))
    return pts.astype(np.float64), v


def run_tracking(sh, affine, pve, seeds, cfg, seed=0, threads=1, tracker="pft"):
    """sh: (n_coef, X, Y, Z) fODF; pve: (3, X, Y, Z) wm/gm/csf on the same grid. Returns a list of (n, 3) arrays in mm.
    tracker: pft (particle filtering: backs up and retries when a path stops inside white matter or CSF) or prob
    (local probabilistic: such paths are dropped). Both use the same seeds, step, angle and tissue-based stopping,
    and keep only streamlines that end in grey matter."""
    t = dict(cfg.tracking)
    if tracker == "prob" and sys.platform == "darwin":
        # The macOS wheels of dipy and PyTorch each load their own OpenMP runtime, and dipy's threaded tracker
        # crashes once both are in one process. A fresh interpreter that never imports torch avoids that.
        with tempfile.TemporaryDirectory() as d:
            np.savez(f"{d}/in.npz", sh=sh, affine=affine, pve=pve, seeds=seeds, settings=json.dumps(t), seed=seed, threads=threads)
            subprocess.run([sys.executable, "-m", "retinopath.track", f"{d}/in.npz", f"{d}/out.npz"], check=True)
            z = np.load(f"{d}/out.npz")
            return [z["points"][i:j] for i, j in zip(z["offsets"][:-1], z["offsets"][1:])]
    return _track(sh, affine, pve, seeds, t, seed, threads, tracker)


def _track(sh, affine, pve, seeds, t, seed, threads, tracker):
    vox = float(np.abs(np.diag(affine)[:3]).mean())
    sc = CmcStoppingCriterion.from_pve(pve[0].astype(float), pve[1].astype(float), pve[2].astype(float),
                                       step_size=t["step_mm"], average_voxel_size=vox)
    kw = dict(sh=np.ascontiguousarray(np.moveaxis(sh, 0, -1), dtype=float), sphere=tracking_sphere(), step_size=t["step_mm"],
              max_angle=t["max_angle"], min_len=t["min_len_mm"], max_len=t["max_len_mm"], return_all=False,
              random_seed=seed + 1, nbr_threads=threads)
    if tracker == "prob":
        gen = probabilistic_tracking(seeds, sc, affine.astype(float), **kw)
    else:
        gen = pft_tracking(seeds, sc, affine.astype(float), pft_back_tracking_dist=t["pft_back_mm"],
                           pft_front_tracking_dist=t["pft_front_mm"], particle_count=t["particles"], **kw)
    return [np.asarray(s, dtype=np.float32) for s in gen]


def endpoints(streamlines, anat, cfg):
    """For every streamline: nearest white-surface vertex of each end (index into anat arrays, -1 if farther than
    the tolerance) and its length in mm."""
    if not len(streamlines):
        z = np.zeros(0, int); return dict(v0=z, v1=z, length=np.zeros(0, np.float32))
    tree = cKDTree(anat["verts"])
    ends = np.stack([np.stack([s[0], s[-1]]) for s in streamlines])                    # (n, 2, 3)
    d, v = tree.query(ends.reshape(-1, 3), distance_upper_bound=cfg.tracking.endpoint_tol_mm)
    v = np.where(np.isfinite(d), v, -1).reshape(-1, 2)
    length = np.array([np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in streamlines], np.float32)
    return dict(v0=v[:, 0], v1=v[:, 1], length=length)


def area_pairs(ep, anat):
    """Keep streamlines whose ends lie in two different areas among V1, V2, V3 of the same hemisphere.
    Returns vertex indices a (lower area), b (higher area) and the pair code 12, 23 or 13."""
    ok = (ep["v0"] >= 0) & (ep["v1"] >= 0)
    v0, v1 = ep["v0"][ok], ep["v1"][ok]
    a0, a1 = anat["area"][v0], anat["area"][v1]
    keep = (a0 > 0) & (a1 > 0) & (a0 != a1) & (anat["hemi"][v0] == anat["hemi"][v1])
    v0, v1, a0, a1 = v0[keep], v1[keep], a0[keep], a1[keep]
    swap = a0 > a1
    a, b = np.where(swap, v1, v0), np.where(swap, v0, v1)
    return a, b, (np.minimum(a0, a1) * 10 + np.maximum(a0, a1)).astype(int), np.flatnonzero(ok)[keep]


def density_map(streamlines, affine, shape):
    """Streamline count per voxel (each streamline counted once per voxel)."""
    import nibabel as nib
    d = np.zeros(shape, np.float32); inv = np.linalg.inv(affine)
    for s in streamlines:
        ijk = np.unique(np.rint(nib.affines.apply_affine(inv, s)).astype(int), axis=0)
        ijk = ijk[np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)]
        d[tuple(ijk.T)] += 1
    return d


if __name__ == "__main__":                                    # python -m retinopath.track in.npz out.npz (see run_tracking)
    z = np.load(sys.argv[1])
    sl = _track(z["sh"], z["affine"], z["pve"], z["seeds"], json.loads(str(z["settings"])), int(z["seed"]), int(z["threads"]), "prob")
    np.savez(sys.argv[2], points=np.concatenate(sl) if sl else np.zeros((0, 3), np.float32), offsets=np.cumsum([0] + [len(x) for x in sl]))
