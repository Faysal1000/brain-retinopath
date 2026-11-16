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
    tracking = cfg.tracking
    rng = np.random.default_rng(seed)
    seed_vertices = np.flatnonzero(anat["area"] > 0).repeat(cfg.seeds_per_vertex)
    points = (
        anat["verts"][seed_vertices]
        - tracking.seed_depth_mm * anat["normals"][seed_vertices]
        + rng.normal(0, tracking.seed_jitter_mm, (len(seed_vertices), 3))
    )
    return points.astype(np.float64), seed_vertices


def run_tracking(sh, affine, pve, seeds, cfg, seed=0, threads=1, tracker="pft"):
    """sh: (n_coef, X, Y, Z) fODF; pve: (3, X, Y, Z) on the same grid. Returns (n, 3) streamlines in mm.
    tracker: pft (particle filtering) or prob (local probabilistic), same seeds and settings."""
    settings = dict(cfg.tracking)
    if tracker == "prob" and sys.platform == "darwin":
        # separate process without torch (two OpenMP runtimes crash dipy's threaded tracker on macOS)
        with tempfile.TemporaryDirectory() as folder:
            np.savez(
                f"{folder}/in.npz",
                sh=sh,
                affine=affine,
                pve=pve,
                seeds=seeds,
                settings=json.dumps(settings),
                seed=seed,
                threads=threads,
            )
            subprocess.run(
                [sys.executable, "-m", "retinopath.track", f"{folder}/in.npz", f"{folder}/out.npz"], check=True
            )
            result = np.load(f"{folder}/out.npz")
            return [result["points"][i:j] for i, j in zip(result["offsets"][:-1], result["offsets"][1:])]
    return _track(sh, affine, pve, seeds, settings, seed, threads, tracker)


def _track(sh, affine, pve, seeds, settings, seed, threads, tracker):
    voxel_mm = float(np.abs(np.diag(affine)[:3]).mean())
    stopping = CmcStoppingCriterion.from_pve(
        pve[0].astype(float),
        pve[1].astype(float),
        pve[2].astype(float),
        step_size=settings["step_mm"],
        average_voxel_size=voxel_mm,
    )
    options = dict(
        sh=np.ascontiguousarray(np.moveaxis(sh, 0, -1), dtype=float),
        sphere=tracking_sphere(),
        step_size=settings["step_mm"],
        max_angle=settings["max_angle"],
        pmf_threshold=settings["pmf_threshold"],
        min_len=settings["min_len_mm"],
        max_len=settings["max_len_mm"],
        return_all=False,
        random_seed=seed + 1,
        nbr_threads=threads,
    )
    if tracker == "prob":
        streamlines = probabilistic_tracking(seeds, stopping, affine.astype(float), **options)
    else:
        streamlines = pft_tracking(
            seeds,
            stopping,
            affine.astype(float),
            pft_back_tracking_dist=settings["pft_back_mm"],
            pft_front_tracking_dist=settings["pft_front_mm"],
            particle_count=settings["particles"],
            **options,
        )
    return [np.asarray(streamline, dtype=np.float32) for streamline in streamlines]


def endpoints(streamlines, anat, cfg):
    """For every streamline: nearest white-surface vertex of each end (index into anat arrays, -1 if farther than
    the tolerance) and its length in mm."""
    if not len(streamlines):
        empty = np.zeros(0, int)
        return dict(v0=empty, v1=empty, length=np.zeros(0, np.float32))
    tree = cKDTree(anat["verts"])
    ends = np.stack([np.stack([streamline[0], streamline[-1]]) for streamline in streamlines])  # (n, 2, 3)
    distance, vertex = tree.query(ends.reshape(-1, 3), distance_upper_bound=cfg.tracking.endpoint_tol_mm)
    vertex = np.where(np.isfinite(distance), vertex, -1).reshape(-1, 2)
    length = np.array(
        [np.linalg.norm(np.diff(streamline, axis=0), axis=1).sum() for streamline in streamlines], np.float32
    )
    return dict(v0=vertex[:, 0], v1=vertex[:, 1], length=length)


def area_pairs(ends, anat):
    """Keep streamlines whose ends lie in two different areas among V1, V2, V3 of the same hemisphere.
    Returns vertex indices a (lower area), b (higher area) and the pair code 12, 23 or 13."""
    on_cortex = (ends["v0"] >= 0) & (ends["v1"] >= 0)
    first, second = ends["v0"][on_cortex], ends["v1"][on_cortex]
    area_first, area_second = anat["area"][first], anat["area"][second]
    keep = (
        (area_first > 0)
        & (area_second > 0)
        & (area_first != area_second)
        & (anat["hemi"][first] == anat["hemi"][second])
    )
    first, second, area_first, area_second = first[keep], second[keep], area_first[keep], area_second[keep]
    swap = area_first > area_second
    lower, higher = np.where(swap, second, first), np.where(swap, first, second)
    return (
        lower,
        higher,
        (np.minimum(area_first, area_second) * 10 + np.maximum(area_first, area_second)).astype(int),
        np.flatnonzero(on_cortex)[keep],
    )


def density_map(streamlines, affine, shape):
    """Streamline count per voxel (each streamline counted once per voxel)."""
    import nibabel as nib

    density = np.zeros(shape, np.float32)
    to_voxel = np.linalg.inv(affine)
    for streamline in streamlines:
        voxels = np.unique(np.rint(nib.affines.apply_affine(to_voxel, streamline)).astype(int), axis=0)
        voxels = voxels[np.all((voxels >= 0) & (voxels < np.array(shape)), axis=1)]
        density[tuple(voxels.T)] += 1
    return density


if __name__ == "__main__":  # python -m retinopath.track in.npz out.npz (see run_tracking)
    z = np.load(sys.argv[1])
    streamlines = _track(
        z["sh"],
        z["affine"],
        z["pve"],
        z["seeds"],
        json.loads(str(z["settings"])),
        int(z["seed"]),
        int(z["threads"]),
        "prob",
    )
    np.savez(
        sys.argv[2],
        points=np.concatenate(streamlines) if streamlines else np.zeros((0, 3), np.float32),
        offsets=np.cumsum([0] + [len(x) for x in streamlines]),
    )
