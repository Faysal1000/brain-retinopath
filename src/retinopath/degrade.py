"""Turn a full-quality HCP crop into a lower-quality acquisition: fewer shells/directions, coarser voxels, noise."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import convolve1d


def pick_directions(bvecs, n):
    """Greedy farthest-point subset of n directions (antipodally symmetric), deterministic."""
    if n == "all" or n >= len(bvecs):
        return np.arange(len(bvecs))
    cos = np.abs(bvecs @ bvecs.T)
    chosen = [0]; closest = cos[0].copy()
    for _ in range(n - 1):
        closest[chosen] = np.inf
        nxt = int(np.argmin(closest)); chosen.append(nxt)
        closest = np.maximum(closest, cos[nxt])
    return np.sort(np.array(chosen))


def select_volumes(bvals, bvecs, shells, n_b0):
    """Volume indices: the first n_b0 b=0 volumes, then for each shell {b: n} the pick_directions subset."""
    idx = list(np.flatnonzero(bvals == 0)[:n_b0])
    for b, n in shells.items():
        s = np.flatnonzero(bvals == int(b))
        idx += list(s[pick_directions(bvecs[s], n)])
    return np.array(idx)


def resample(vol, affine, voxel_out, lowpass=True):
    """Low-pass then trilinear resample a (X, Y, Z[, N]) volume to a coarser isotropic grid. Returns vol, affine."""
    vin = float(abs(affine[0, 0]))
    if abs(voxel_out - vin) < 1e-3:
        return vol, affine
    v = np.asarray(vol, np.float32)
    sigma = np.sqrt(max(voxel_out ** 2 - vin ** 2, 0)) / 2.355 / vin        # input voxels; FWHM = sqrt(out^2 - in^2) mm
    if sigma > 0 and lowpass:
        # separable Gaussian, edges repeated. SciPy rather than torch.conv3d: the CPU convolution of the Linux aarch64
        # PyTorch wheel (oneDNN) fails on these volumes with "label is too far"
        r = int(np.ceil(3 * sigma)); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
        for d in range(3):
            v = convolve1d(v, k.astype(np.float32), axis=d, mode="nearest")
    x = torch.from_numpy(np.ascontiguousarray(v))
    x = x[..., None] if x.ndim == 3 else x
    x = x.permute(3, 0, 1, 2)[:, None]                                      # (N, 1, X, Y, Z)
    size = [max(2, int(round(s * vin / voxel_out))) for s in x.shape[2:]]
    y = F.interpolate(x, size=size, mode="trilinear", align_corners=False)[:, 0].permute(1, 2, 3, 0).numpy()
    s = np.array(x.shape[2:]) / np.array(size)                              # exact per-axis scale
    m = np.eye(4); m[:3, :3] = np.diag(s); m[:3, 3] = (s - 1) / 2          # voxel-centre shift for align_corners=False
    return (y[..., 0] if vol.ndim == 3 else y), affine @ m


def to_grid(vol, shape):
    """Trilinear resample of (X, Y, Z[, N]) back onto a grid that shares the same field of view."""
    x = torch.from_numpy(np.ascontiguousarray(vol)).float()
    x = x[..., None] if x.ndim == 3 else x
    y = F.interpolate(x.permute(3, 0, 1, 2)[None], size=tuple(int(s) for s in shape), mode="trilinear",
                      align_corners=False)[0].permute(1, 2, 3, 0).numpy()
    return y[..., 0] if vol.ndim == 3 else y


def add_rician(dwi, bvals, snr, wm, rng):
    """Rician noise with sigma = mean b=0 signal in white matter / snr. White matter = fraction above 0.9, or the
    purest 5% of voxels when a coarse grid has none that pure."""
    sigma = dwi[..., bvals == 0].mean(-1)[wm > min(0.9, np.quantile(wm, 0.95))].mean() / snr
    return np.sqrt((dwi + rng.normal(0, sigma, dwi.shape)) ** 2 + rng.normal(0, sigma, dwi.shape) ** 2).astype(np.float32)


def make_protocol(dwi, bvals, bvecs, affine, pve, spec, n_b0=3, seed=0):
    """Return dict(dwi, bvals, bvecs, affine, pve) for one protocol spec from config.yaml."""
    idx = select_volumes(bvals, bvecs, spec["shells"], n_b0)
    d, bv, bc = dwi[..., idx], bvals[idx], bvecs[idx]
    d, aff = resample(d, affine, spec["voxel"])
    p = np.stack([resample(m, affine, spec["voxel"])[0] for m in pve])
    if spec.get("snr"):               # after resampling: noise belongs to the acquired voxels, averaging would remove it
        d = add_rician(d, bv, spec["snr"], p[0], np.random.default_rng(seed))
    return dict(dwi=d, bvals=bv, bvecs=bc, affine=aff.astype(np.float32), pve=p)
