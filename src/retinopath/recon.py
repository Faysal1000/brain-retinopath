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
    avoided because it segfaults on macOS after dipy's OpenMP code), torch float64 on CUDA (_admm_torch)."""
    X, C = np.asarray(X, np.float64), np.asarray(C, np.float64)
    P = X.T @ X
    rho = rho or float(np.trace(P) / np.trace(C.T @ C))
    Minv = np.linalg.inv(P + rho * C.T @ C)
    if str(device).startswith("cuda"):
        return np.concatenate([_admm_torch(S[i:i + 50_000], X, C, Minv, rho, iters, device) for i in range(0, len(S), 50_000)])
    out = []
    for i in range(0, len(S), 50_000):
        s = np.asarray(S[i:i + 50_000], np.float64); q = s @ X; z = np.zeros((len(s), C.shape[0])); u = np.zeros((len(s), C.shape[0]))
        for _ in range(iters):
            x = (q + rho * (z - u) @ C) @ Minv
            cx = x @ C.T
            z = np.maximum(cx + u, 0); u = u + cx - z
        out.append(x.astype(np.float32))
    return np.concatenate(out)


def _admm_torch(S, X, C, Minv, rho, iters, device):
    """The iterations of solve_qp_batch for one chunk of rows, in torch float64, without a new array per step.
    With t = C x + u the updates are u = min(t, 0) and z - u = |t|, exactly; v holds rho (z - u)."""
    import torch
    to = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64, device=device)
    X, C, Minv = to(X), to(C), to(Minv); Ct = C.T.contiguous()
    q = to(S) @ X
    v, u, t = (torch.zeros(len(q), C.shape[0], dtype=torch.float64, device=device) for _ in range(3))
    for _ in range(iters):
        x = (q + v @ C) @ Minv
        torch.matmul(x, Ct, out=t); t.add_(u)
        torch.clamp(t, max=0, out=u); t.abs_().mul_(rho)
        v, t = t, v
    return x.float().cpu().numpy()


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


def _bessel_ratio(x):
    """I1(x) / I0(x) by the continued fraction dipy's RUMBA-SD uses (rumba.mbessel_ratio with n = 1)."""
    return x / ((2 + x) - (3 * x / (3 + 2 * x - (5 * x / (4 + 2 * x - (7 * x / (5 + 2 * x)))))))


def rumba_batch(D, K, n_iter, device="cpu"):
    """RUMBA-SD for many voxels at once: the Richardson-Lucy iterations of dipy's voxel-wise fit (rumba_deconv,
    Rician noise, noise variance re-estimated each iteration) written for all rows of D together. D: (n, n_grad)
    signal divided by b=0, first column 1; K: (n_grad, n_comp) kernel. Returns (n, n_comp) fractions that sum to 1.
    Agrees with dipy to rounding error (1e-15). NumPy float64 on the CPU, torch float64 on CUDA; rows in chunks
    of 20,000."""
    eps = np.finfo(float).eps; n = K.shape[1]; lo, hi = (1 / 80) ** 2, (1 / 8) ** 2
    if str(device).startswith("cuda"):
        import torch
        K = torch.as_tensor(K, dtype=torch.float64, device=device); Kt = K.T.contiguous(); out = []
        for i in range(0, len(D), 20_000):
            d = torch.as_tensor(np.asarray(D[i:i + 20_000]), dtype=torch.float64, device=device)
            f = torch.full((len(d), n), 1.0 / n, dtype=torch.float64, device=device)
            reb = f @ Kt; s2 = torch.full((len(d), 1), (1 / 15) ** 2, dtype=torch.float64, device=device); rs = d * reb / s2
            for _ in range(n_iter):
                ratio = _bessel_ratio(rs)
                f = (f * (((d * ratio) @ K) / (reb @ K + eps))).clamp_(min=0)
                reb = f @ Kt; rs = d * reb / s2
                s2 = (((d ** 2 + reb ** 2) / 2 - (s2 * rs) * ratio).sum(1, keepdim=True) / d.shape[1]).clamp_(lo, hi)
            out.append((f / (f.sum(1, keepdim=True) + eps)).cpu().numpy())
        return np.concatenate(out)
    K = np.asarray(K, np.float64); out = []
    for i in range(0, len(D), 20_000):
        d = np.asarray(D[i:i + 20_000], np.float64); f = np.full((len(d), n), 1.0 / n)
        reb = f @ K.T; s2 = np.full((len(d), 1), (1 / 15) ** 2); rs = d * reb / s2
        for _ in range(n_iter):
            ratio = _bessel_ratio(rs)
            f = np.maximum(f * (((d * ratio) @ K) / (reb @ K + eps)), 0)
            reb = f @ K.T; rs = d * reb / s2
            s2 = np.clip(((d ** 2 + reb ** 2) / 2 - (s2 * rs) * ratio).sum(1, keepdims=True) / d.shape[1], lo, hi)
        out.append(f / (f.sum(1, keepdims=True) + eps))
    return np.concatenate(out)


def fit_rumba(p, mask, jobs=1, order=8, n_iter=200, device="cpu"):
    """RUMBA-SD (Canales-Rodriguez et al. 2015) with dipy's kernel and settings, solved for all voxels together
    (rumba_batch). The fODF on a sphere is converted to SH so every method tracks alike."""
    from dipy.reconst.rumba import RumbaSDModel, generate_kernel
    gtab = gtab_of(p); sph = get_sphere(name="symmetric362")
    flat = p["dwi"].reshape(-1, p["dwi"].shape[-1])
    ix = _top_fa(p, gtab, p["pve"][0]); fit = TensorModel(gtab).fit(flat[ix])
    wm_resp = np.array([fit.evals[:, 0].mean(), fit.evals[:, 1:].mean(), fit.evals[:, 1:].mean()])
    model = RumbaSDModel(gtab, wm_response=wm_resp, n_iter=n_iter, sphere=sph, voxelwise=True)
    K = generate_kernel(model.gtab, sph, wm_resp, model.gm_response, model.csf_response)
    inv = np.linalg.pinv(sh_matrix(sph.vertices, order))

    def run(ix):
        d = shm.normalize_data(flat[ix].astype(np.float64), model.where_b0s, min_signal=np.finfo(float).eps)
        d = np.concatenate([np.ones((len(d), 1)), d[:, model.where_dwi]], 1); d[d > 1] = 1
        return rumba_batch(d, K, n_iter, device)[:, :-2] @ inv.T

    if str(device).startswith("cuda"):
        chunks = [np.flatnonzero(mask.ravel())]; parts = [run(chunks[0])]
    else:
        chunks = _chunks(mask, jobs * 4); parts = Parallel(n_jobs=jobs)(delayed(run)(ix) for ix in chunks)
    out = np.zeros((mask.size, n_coef(order)), np.float32)
    for ix, c in zip(chunks, parts):
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
        c = fit_rumba(p, mask, jobs, n_iter=cfg.rumba_iters, device=device)
    else:
        c = fit_csd(p, mask, jobs)
    return pad_order(c, cfg.sh_order_out)
