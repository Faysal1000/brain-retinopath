"""Spherical harmonics helpers and classical fODF estimation (CSD, multi-shell multi-tissue CSD, RUMBA-SD)."""
import warnings
import numpy as np
from dipy.core.gradients import gradient_table
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from dipy.reconst import shm
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.dti import TensorModel
from joblib import Parallel, delayed


def n_coef(order):
    return (order + 1) * (order + 2) // 2


def sh_matrix(dirs, order):
    """(n_dirs, n_coef) matrix evaluating a real even SH series at unit vectors `dirs`."""
    s = Sphere(xyz=np.asarray(dirs, float))
    return shm.real_sh_descoteaux(order, s.theta, s.phi, legacy=True)[0].astype(np.float32)


def tracking_sphere():
    return get_sphere(name="symmetric724")


def max_order(n_dirs, cap=8):
    """Highest even SH order <= cap with at least 1.3 directions per coefficient (minimum 2)."""
    for o in (8, 6, 4, 2):
        if o <= cap and n_dirs >= n_coef(o) * 1.3:
            return o
    return 2


def signal_sh(dwi, bvals, bvecs, shells, order=4, lam=0.006):
    """Per-shell SH coefficients of the b0-normalised signal: (len(shells)*n_coef, X, Y, Z). Missing shells = 0."""
    b0 = dwi[..., bvals == 0].mean(-1); b0s = np.maximum(b0, 1e-3)
    nc = n_coef(order); out = np.zeros((len(shells) * nc,) + dwi.shape[:3], np.float32)
    for i, b in enumerate(shells):
        s = bvals == b
        if not s.any():
            continue
        o = min(order, max_order(int(s.sum()), order))
        B = sh_matrix(bvecs[s], o); l = np.concatenate([[k] * (2 * k + 1) for k in range(0, o + 1, 2)])
        L = np.diag((l * (l + 1)) ** 2).astype(np.float32)
        pinv = np.linalg.solve(B.T @ B + lam * L, B.T)                       # (nc_o, n_dirs)
        c = np.clip(dwi[..., s] / b0s[..., None], 0, 2) @ pinv.T
        out[i * nc:i * nc + c.shape[-1]] = np.moveaxis(c, -1, 0)
    return out * (b0 > 0)


def gtab_of(p):
    return gradient_table(p["bvals"].astype(float), bvecs=p["bvecs"], b0_threshold=50)


def _chunks(mask, n):
    idx = np.flatnonzero(mask.ravel())
    return [c for c in np.array_split(idx, max(1, n)) if len(c)]


def _fit_parallel(model, data, mask, jobs, attr="shm_coeff"):
    """Fit `model` on masked voxels in parallel; returns (n_coef, X, Y, Z) float32."""
    flat = data.reshape(-1, data.shape[-1])

    def run(ix):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return getattr(model.fit(flat[ix]), attr)

    parts = Parallel(n_jobs=jobs)(delayed(run)(ix) for ix in _chunks(mask, jobs * 4))
    nc = parts[0].shape[-1]; out = np.zeros((mask.size, nc), np.float32)
    for ix, p in zip(_chunks(mask, jobs * 4), parts):
        out[ix] = p
    return np.moveaxis(out.reshape(mask.shape + (nc,)), -1, 0)


def _top_fa(p, gtab, wm, n=300):
    """Flat indices of the n highest-FA voxels with wm > 0.95 (single-fibre voxels for response functions)."""
    low = gtab.bvals <= max(1500, gtab.bvals[~gtab.b0s_mask].min() + 100)        # tensor fit on the lowest shell
    g = gradient_table(gtab.bvals[low], bvecs=gtab.bvecs[low], b0_threshold=50)
    ix = np.flatnonzero((wm > 0.95).ravel())
    fit = TensorModel(g).fit(p["dwi"].reshape(-1, p["dwi"].shape[-1])[ix][:, low])
    return ix[np.argsort(fit.fa)[-min(n, len(ix)):]]


def fit_csd(p, mask, jobs=1, order=None):
    """Single-shell CSD on the highest b-value shell. Returns white-matter fODF SH (n_coef(order), X, Y, Z)."""
    b = p["bvals"]; keep = (b == 0) | (b == b.max())
    q = dict(p, dwi=p["dwi"][..., keep], bvals=b[keep], bvecs=p["bvecs"][keep]); gtab = gtab_of(q)
    order = order or max_order(int((q["bvals"] > 0).sum()))
    ix = _top_fa(q, gtab, q["pve"][0])
    vox = q["dwi"].reshape(-1, q["dwi"].shape[-1])[ix]
    fit = TensorModel(gtab).fit(vox)
    response = (fit.evals.mean(0) * [1, 0, 0] + fit.evals[:, 1:].mean() * np.array([0, 1, 1]), vox[:, gtab.b0s_mask].mean())
    return _fit_parallel(ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=order), q["dwi"], mask, jobs)


def solve_qp_batch(X, C, S, iters=1000, rho=None, device="cpu"):
    """Batched ADMM for  min ||X x - s||^2  subject to  C x >= 0,  one problem per row of S (the constrained fit
    dipy's MSMT-CSD solves voxel by voxel with cvxpy). Returns (n, n_params) float32.

    Splitting z = C x with z >= 0 and scaled dual u, each iteration is
        x = (X's + rho C'(z - u)) (X'X + rho C'C)^-1,   z = max(C x + u, 0),   u = u + C x - z
    with the inverse computed once. Fixed number of iterations, no convergence test. rho defaults to
    tr(X'X) / tr(C'C). Rows are processed in chunks of 50,000. NumPy float64 on the CPU (torch CPU maths is
    avoided because it segfaults on macOS after dipy's OpenMP code), torch float64 on CUDA."""
    X, C = np.asarray(X, np.float64), np.asarray(C, np.float64)
    P = X.T @ X
    rho = rho or float(np.trace(P) / np.trace(C.T @ C))
    Minv = np.linalg.inv(P + rho * C.T @ C)
    if str(device).startswith("cuda"):
        import torch
        to = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
        X, C, Minv, xp = to(X), to(C), to(Minv), torch
        relu, zeros, back = (lambda a: torch.clamp(a, min=0)), (lambda n: torch.zeros(n, C.shape[0], dtype=torch.float64, device=device)), (lambda a: a.float().cpu().numpy())
    else:
        to, relu, zeros, back = (lambda a: np.asarray(a, np.float64)), (lambda a: np.maximum(a, 0)), (lambda n: np.zeros((n, C.shape[0]))), (lambda a: a.astype(np.float32))
    out = []
    for i in range(0, len(S), 50_000):
        s = to(S[i:i + 50_000]); q = s @ X; z = zeros(len(s)); u = zeros(len(s))
        for _ in range(iters):
            x = (q + rho * (z - u) @ C) @ Minv
            cx = x @ C.T
            z = relu(cx + u); u = u + cx - z
        out.append(back(x))
    return np.concatenate(out)


def fit_msmt(p, mask, jobs=1, order=8, device="cpu", use_dipy=False):
    """Multi-shell multi-tissue CSD (needs >= 2 shells). Returns white-matter fODF SH.
    dipy builds the response functions and matrices; the constrained fit is solved for all voxels at once."""
    from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response, response_from_mask_msmt
    gtab = gtab_of(p); flat_n = p["dwi"].shape[:3]
    wm = np.zeros(flat_n, bool); wm.ravel()[_top_fa(p, gtab, p["pve"][0])] = True
    b0 = p["dwi"][..., gtab.b0s_mask].mean(-1); ok = b0 > 0.2 * np.median(b0[p["pve"][0] > 0.9])
    gm = (p["pve"][1] > 0.97) & ok; csf = (p["pve"][2] > 0.97) & ok
    rw, rg, rc = response_from_mask_msmt(gtab, p["dwi"], wm, gm, csf)
    ub = np.concatenate([[0], np.unique(gtab.bvals[~gtab.b0s_mask])])
    model = MultiShellDeconvModel(gtab, multi_shell_fiber_response(order, ub, rw, rg, rc), sh_order_max=order)
    if use_dipy:                                               # dipy's per-voxel cvxpy fit: slow, for checking solve_qp_batch
        return _fit_parallel(model, p["dwi"], mask, jobs)
    x = solve_qp_batch(model._X, model.fitter._reg, p["dwi"][mask], device=device)
    out = np.zeros((n_coef(order),) + flat_n, np.float32)
    out[:, mask] = x[:, 2:].T                                  # drop the two isotropic (CSF, GM) compartments
    return out


def fit_rumba(p, mask, jobs=1, order=8, n_iter=200):
    """RUMBA-SD (Canales-Rodriguez et al. 2015). The fODF on a sphere is converted to SH so every method tracks alike."""
    from dipy.reconst.rumba import RumbaSDModel
    gtab = gtab_of(p); sph = get_sphere(name="symmetric362")
    ix = _top_fa(p, gtab, p["pve"][0]); fit = TensorModel(gtab).fit(p["dwi"].reshape(-1, p["dwi"].shape[-1])[ix])
    wm_resp = np.array([fit.evals[:, 0].mean(), fit.evals[:, 1:].mean(), fit.evals[:, 1:].mean()])
    model = RumbaSDModel(gtab, wm_response=wm_resp, n_iter=n_iter, sphere=sph, voxelwise=True)
    flat = p["dwi"].reshape(-1, p["dwi"].shape[-1]); inv = np.linalg.pinv(sh_matrix(sph.vertices, order))

    def run(ix):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit(flat[ix]).odf() @ inv.T

    parts = Parallel(n_jobs=jobs)(delayed(run)(ix) for ix in _chunks(mask, jobs * 4))
    out = np.zeros((mask.size, n_coef(order)), np.float32)
    for ix, c in zip(_chunks(mask, jobs * 4), parts):
        out[ix] = c
    return np.moveaxis(out.reshape(mask.shape + (n_coef(order),)), -1, 0)


def pad_order(coef, order=8):
    """Zero-pad a lower-order SH volume to `order` so every method has the same number of channels."""
    out = np.zeros((n_coef(order),) + coef.shape[1:], np.float32); out[:coef.shape[0]] = coef; return out


def fit_fodf(p, mask, method, cfg, jobs=1, device="cpu"):
    """method: msmt_csd (single-shell CSD if the data have one shell), rumba, anything else = csd.
    Returns (n_coef(sh_order_out), X, Y, Z), zero-padded to sh_order_out."""
    multi = len(np.unique(p["bvals"][p["bvals"] > 0])) > 1
    if method == "msmt_csd" and multi:
        c = fit_msmt(p, mask, jobs, device=device)
    elif method == "rumba":
        c = fit_rumba(p, mask, jobs, n_iter=cfg.rumba_iters)
    else:
        c = fit_csd(p, mask, jobs)
    return pad_order(c, cfg.sh_order_out)
