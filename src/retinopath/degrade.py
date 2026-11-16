"""Turn a full-quality HCP crop into a lower-quality acquisition: fewer shells/directions, coarser voxels, noise."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve1d


def pick_directions(bvecs, count):
    """Greedy farthest-point subset of n directions (antipodally symmetric), deterministic."""
    if count == "all" or count >= len(bvecs):
        return np.arange(len(bvecs))
    cosine = np.abs(bvecs @ bvecs.T)
    chosen = [0]
    closest = cosine[0].copy()
    for _ in range(count - 1):
        closest[chosen] = np.inf
        farthest = int(np.argmin(closest))
        chosen.append(farthest)
        closest = np.maximum(closest, cosine[farthest])
    return np.sort(np.array(chosen))


def select_volumes(bvals, bvecs, shells, n_b0):
    """Volume indices: the first n_b0 b=0 volumes, then for each shell {b: n} the pick_directions subset."""
    indices = list(np.flatnonzero(bvals == 0)[:n_b0])
    for bvalue, count in shells.items():
        shell = np.flatnonzero(bvals == int(bvalue))
        indices += list(shell[pick_directions(bvecs[shell], count)])
    return np.array(indices)


def resample(vol, affine, voxel_out, lowpass=True):
    """Low-pass then trilinear resample a (X, Y, Z[, N]) volume to a coarser isotropic grid. Returns vol, affine."""
    voxel_in = float(abs(affine[0, 0]))
    if abs(voxel_out - voxel_in) < 1e-3:
        return vol, affine
    data = np.asarray(vol, np.float32)
    sigma = np.sqrt(max(voxel_out**2 - voxel_in**2, 0)) / 2.355 / voxel_in  # input voxels; FWHM = sqrt(out^2 - in^2) mm
    if sigma > 0 and lowpass:
        # separable Gaussian, edges repeated (SciPy: torch's CPU conv3d fails on Linux aarch64)
        radius = int(np.ceil(3 * sigma))
        kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
        kernel /= kernel.sum()
        for axis in range(3):
            data = convolve1d(data, kernel.astype(np.float32), axis=axis, mode="nearest")
    tensor = torch.from_numpy(np.ascontiguousarray(data))
    tensor = tensor[..., None] if tensor.ndim == 3 else tensor
    tensor = tensor.permute(3, 0, 1, 2)[:, None]  # (N, 1, X, Y, Z)
    size = [max(2, int(round(length * voxel_in / voxel_out))) for length in tensor.shape[2:]]
    resampled = (
        F.interpolate(tensor, size=size, mode="trilinear", align_corners=False)[:, 0].permute(1, 2, 3, 0).numpy()
    )
    scale = np.array(tensor.shape[2:]) / np.array(size)  # exact per-axis scale
    scaling = np.eye(4)
    scaling[:3, :3] = np.diag(scale)
    scaling[:3, 3] = (scale - 1) / 2  # voxel-centre shift for align_corners=False
    return (resampled[..., 0] if vol.ndim == 3 else resampled), affine @ scaling


def to_grid(vol, shape):
    """Trilinear resample of (X, Y, Z[, N]) back onto a grid that shares the same field of view."""
    tensor = torch.from_numpy(np.ascontiguousarray(vol)).float()
    tensor = tensor[..., None] if tensor.ndim == 3 else tensor
    resampled = (
        F.interpolate(
            tensor.permute(3, 0, 1, 2)[None],
            size=tuple(int(length) for length in shape),
            mode="trilinear",
            align_corners=False,
        )[0]
        .permute(1, 2, 3, 0)
        .numpy()
    )
    return resampled[..., 0] if vol.ndim == 3 else resampled


def add_rician(dwi, bvals, snr, wm, rng):
    """Rician noise, sigma = mean white-matter b=0 signal / snr (wm > 0.9, or the purest 5% of voxels)."""
    sigma = dwi[..., bvals == 0].mean(-1)[wm > min(0.9, np.quantile(wm, 0.95))].mean() / snr
    return np.sqrt((dwi + rng.normal(0, sigma, dwi.shape)) ** 2 + rng.normal(0, sigma, dwi.shape) ** 2).astype(
        np.float32
    )


def make_protocol(dwi, bvals, bvecs, affine, pve, spec, n_b0=3, seed=0):
    """Return dict(dwi, bvals, bvecs, affine, pve) for one protocol spec from config.yaml."""
    indices = select_volumes(bvals, bvecs, spec["shells"], n_b0)
    signal, kept_bvals, kept_bvecs = dwi[..., indices], bvals[indices], bvecs[indices]
    signal, new_affine = resample(signal, affine, spec["voxel"])
    tissue = np.stack([resample(tissue_map, affine, spec["voxel"])[0] for tissue_map in pve])
    if spec.get("snr"):  # after resampling, on the acquired voxels
        signal = add_rician(signal, kept_bvals, spec["snr"], tissue[0], np.random.default_rng(seed))
    return dict(dwi=signal, bvals=kept_bvals, bvecs=kept_bvecs, affine=new_affine.astype(np.float32), pve=tissue)
