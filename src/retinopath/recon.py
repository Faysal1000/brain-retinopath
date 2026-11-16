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
    sphere = Sphere(xyz=np.asarray(dirs, float))
    return shm.real_sh_descoteaux(order, sphere.theta, sphere.phi, legacy=True)[0].astype(np.float32)


def tracking_sphere():
    return get_sphere(name="symmetric724")


def max_order(n_dirs, cap=8):
    """Highest even SH order <= cap with at least 1.3 directions per coefficient (minimum 2)."""
    for order in (8, 6, 4, 2):
        if order <= cap and n_dirs >= n_coef(order) * 1.3:
            return order
    return 2


def signal_sh(dwi, bvals, bvecs, shells, order=4, smoothing=0.006):
    """Per-shell SH coefficients of the b0-normalised signal: (len(shells)*n_coef, X, Y, Z). Missing shells = 0."""
    b0 = dwi[..., bvals == 0].mean(-1)
    b0_floor = np.maximum(b0, 1e-3)
    per_shell = n_coef(order)
    out = np.zeros((len(shells) * per_shell,) + dwi.shape[:3], np.float32)
    for index, bvalue in enumerate(shells):
        in_shell = bvals == bvalue
        if not in_shell.any():
            continue
        shell_order = min(order, max_order(int(in_shell.sum()), order))
        basis = sh_matrix(bvecs[in_shell], shell_order)
        degree = np.concatenate([[even_degree] * (2 * even_degree + 1) for even_degree in range(0, shell_order + 1, 2)])
        laplace = np.diag((degree * (degree + 1)) ** 2).astype(np.float32)
        inverse = np.linalg.solve(basis.T @ basis + smoothing * laplace, basis.T)  # (nc_o, n_dirs)
        coefficients = np.clip(dwi[..., in_shell] / b0_floor[..., None], 0, 2) @ inverse.T
        out[index * per_shell : index * per_shell + coefficients.shape[-1]] = np.moveaxis(coefficients, -1, 0)
    return out * (b0 > 0)


def gtab_of(protocol):
    return gradient_table(protocol["bvals"].astype(float), bvecs=protocol["bvecs"], b0_threshold=50)


def _chunks(mask, count):
    indices = np.flatnonzero(mask.ravel())
    return [chunk for chunk in np.array_split(indices, max(1, count)) if len(chunk)]


def _fit_parallel(model, data, mask, jobs, attr="shm_coeff"):
    """Fit `model` on masked voxels in parallel; returns (n_coef, X, Y, Z) float32."""
    voxels = data.reshape(-1, data.shape[-1])

    def run(indices):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return getattr(model.fit(voxels[indices]), attr)

    parts = Parallel(n_jobs=jobs)(delayed(run)(indices) for indices in _chunks(mask, jobs * 4))
    n_coefficients = parts[0].shape[-1]
    out = np.zeros((mask.size, n_coefficients), np.float32)
    for indices, part in zip(_chunks(mask, jobs * 4), parts):
        out[indices] = part
    return np.moveaxis(out.reshape(mask.shape + (n_coefficients,)), -1, 0)


def _top_fa(protocol, gtab, wm, count=300):
    """Flat indices of the n highest-FA voxels with wm > 0.95 (single-fibre voxels for response functions)."""
    lowest_shell = gtab.bvals <= max(1500, gtab.bvals[~gtab.b0s_mask].min() + 100)  # tensor fit on the lowest shell
    low_gtab = gradient_table(gtab.bvals[lowest_shell], bvecs=gtab.bvecs[lowest_shell], b0_threshold=50)
    candidates = np.flatnonzero((wm > 0.95).ravel())
    tensor_fit = TensorModel(low_gtab).fit(
        protocol["dwi"].reshape(-1, protocol["dwi"].shape[-1])[candidates][:, lowest_shell]
    )
    return candidates[np.argsort(tensor_fit.fa)[-min(count, len(candidates)) :]]


def fit_csd(protocol, mask, jobs=1, order=None):
    """Single-shell CSD on the highest b-value shell. Returns white-matter fODF SH (n_coef(order), X, Y, Z)."""
    bvalues = protocol["bvals"]
    keep = (bvalues == 0) | (bvalues == bvalues.max())
    top_shell = dict(protocol, dwi=protocol["dwi"][..., keep], bvals=bvalues[keep], bvecs=protocol["bvecs"][keep])
    gtab = gtab_of(top_shell)
    order = order or max_order(int((top_shell["bvals"] > 0).sum()))
    single_fibre = _top_fa(top_shell, gtab, top_shell["pve"][0])
    signals = top_shell["dwi"].reshape(-1, top_shell["dwi"].shape[-1])[single_fibre]
    tensor_fit = TensorModel(gtab).fit(signals)
    response = (
        tensor_fit.evals.mean(0) * [1, 0, 0] + tensor_fit.evals[:, 1:].mean() * np.array([0, 1, 1]),
        signals[:, gtab.b0s_mask].mean(),
    )
    return _fit_parallel(
        ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=order), top_shell["dwi"], mask, jobs
    )


def solve_qp_batch(X, C, S, iters=1000, rho=None, device="cpu"):
    """Batched ADMM for min ||X x - s||^2 subject to C x >= 0, one problem per row of S. Returns (n, n_params).

    Split z = C x >= 0 with scaled dual u:
        x = (X's + rho C'(z - u)) (X'X + rho C'C)^-1,   z = max(C x + u, 0),   u = u + C x - z
    Fixed number of iterations; rho defaults to tr(X'X) / tr(C'C). NumPy float64 on the CPU, torch on CUDA."""
    X, C = np.asarray(X, np.float64), np.asarray(C, np.float64)
    gram = X.T @ X
    rho = rho or float(np.trace(gram) / np.trace(C.T @ C))
    inverse = np.linalg.inv(gram + rho * C.T @ C)
    if str(device).startswith("cuda"):
        return np.concatenate(
            [
                _admm_torch(S[start : start + 50_000], X, C, inverse, rho, iters, device)
                for start in range(0, len(S), 50_000)
            ]
        )
    out = []
    for start in range(0, len(S), 50_000):
        signals = np.asarray(S[start : start + 50_000], np.float64)
        projected = signals @ X
        z = np.zeros((len(signals), C.shape[0]))
        u = np.zeros((len(signals), C.shape[0]))
        for _ in range(iters):
            x = (projected + rho * (z - u) @ C) @ inverse
            constraint = x @ C.T
            z = np.maximum(constraint + u, 0)
            u = u + constraint - z
        out.append(x.astype(np.float32))
    return np.concatenate(out)


def _admm_torch(S, X, C, inverse, rho, iters, device):
    """solve_qp_batch iterations for one chunk on CUDA, in place. With t = C x + u: u = min(t, 0), z - u = |t|."""
    import torch

    to_device = lambda array: torch.as_tensor(np.asarray(array), dtype=torch.float64, device=device)
    X, C, inverse = to_device(X), to_device(C), to_device(inverse)
    C_transposed = C.T.contiguous()
    projected = to_device(S) @ X
    v, u, t = (torch.zeros(len(projected), C.shape[0], dtype=torch.float64, device=device) for _ in range(3))
    for _ in range(iters):
        x = (projected + v @ C) @ inverse
        torch.matmul(x, C_transposed, out=t)
        t.add_(u)
        torch.clamp(t, max=0, out=u)
        t.abs_().mul_(rho)
        v, t = t, v
    return x.float().cpu().numpy()


def fit_msmt(protocol, mask, jobs=1, order=8, device="cpu", use_dipy=False):
    """Multi-shell multi-tissue CSD (>= 2 shells), dipy model, batched fit. Returns the white-matter fODF SH."""
    from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response, response_from_mask_msmt

    gtab = gtab_of(protocol)
    shape = protocol["dwi"].shape[:3]
    single_fibre = np.zeros(shape, bool)
    single_fibre.ravel()[_top_fa(protocol, gtab, protocol["pve"][0])] = True
    b0 = protocol["dwi"][..., gtab.b0s_mask].mean(-1)
    bright = b0 > 0.2 * np.median(b0[protocol["pve"][0] > 0.9])
    gm = (protocol["pve"][1] > 0.97) & bright
    csf = (protocol["pve"][2] > 0.97) & bright
    response_wm, response_gm, response_csf = response_from_mask_msmt(gtab, protocol["dwi"], single_fibre, gm, csf)
    unique_bvals = np.concatenate([[0], np.unique(gtab.bvals[~gtab.b0s_mask])])
    model = MultiShellDeconvModel(
        gtab,
        multi_shell_fiber_response(order, unique_bvals, response_wm, response_gm, response_csf),
        sh_order_max=order,
    )
    if use_dipy:  # dipy's per-voxel cvxpy fit: slow, for checking solve_qp_batch
        return _fit_parallel(model, protocol["dwi"], mask, jobs)
    solution = solve_qp_batch(model._X, model.fitter._reg, protocol["dwi"][mask], device=device)
    fodf = np.zeros((n_coef(order),) + shape, np.float32)
    fodf[:, mask] = solution[:, 2:].T  # drop the two isotropic (CSF, GM) compartments
    return fodf


def _bessel_ratio(x):
    """I1(x) / I0(x) by the continued fraction dipy's RUMBA-SD uses (rumba.mbessel_ratio with n = 1)."""
    return x / ((2 + x) - (3 * x / (3 + 2 * x - (5 * x / (4 + 2 * x - (7 * x / (5 + 2 * x)))))))


def rumba_batch(D, K, n_iter, device="cpu"):
    """dipy's voxel-wise RUMBA-SD iterations (rumba_deconv, Rician noise) for all rows at once.
    D: (n, n_grad) signal / b0 with first column 1; K: (n_grad, n_comp) kernel. Returns (n, n_comp) fractions."""
    epsilon = np.finfo(float).eps
    n_components = K.shape[1]
    variance_min, variance_max = (1 / 80) ** 2, (1 / 8) ** 2
    if str(device).startswith("cuda"):
        import torch

        K = torch.as_tensor(K, dtype=torch.float64, device=device)
        K_transposed = K.T.contiguous()
        out = []
        for start in range(0, len(D), 20_000):
            signals = torch.as_tensor(np.asarray(D[start : start + 20_000]), dtype=torch.float64, device=device)
            fractions = torch.full((len(signals), n_components), 1.0 / n_components, dtype=torch.float64, device=device)
            predicted = fractions @ K_transposed
            variance = torch.full((len(signals), 1), (1 / 15) ** 2, dtype=torch.float64, device=device)
            scaled = signals * predicted / variance
            for _ in range(n_iter):
                ratio = _bessel_ratio(scaled)
                fractions = (fractions * (((signals * ratio) @ K) / (predicted @ K + epsilon))).clamp_(min=0)
                predicted = fractions @ K_transposed
                scaled = signals * predicted / variance
                variance = (
                    ((signals**2 + predicted**2) / 2 - (variance * scaled) * ratio).sum(1, keepdim=True)
                    / signals.shape[1]
                ).clamp_(variance_min, variance_max)
            out.append((fractions / (fractions.sum(1, keepdim=True) + epsilon)).cpu().numpy())
        return np.concatenate(out)
    K = np.asarray(K, np.float64)
    out = []
    for start in range(0, len(D), 20_000):
        signals = np.asarray(D[start : start + 20_000], np.float64)
        fractions = np.full((len(signals), n_components), 1.0 / n_components)
        predicted = fractions @ K.T
        variance = np.full((len(signals), 1), (1 / 15) ** 2)
        scaled = signals * predicted / variance
        for _ in range(n_iter):
            ratio = _bessel_ratio(scaled)
            fractions = np.maximum(fractions * (((signals * ratio) @ K) / (predicted @ K + epsilon)), 0)
            predicted = fractions @ K.T
            scaled = signals * predicted / variance
            variance = np.clip(
                ((signals**2 + predicted**2) / 2 - (variance * scaled) * ratio).sum(1, keepdims=True)
                / signals.shape[1],
                variance_min,
                variance_max,
            )
        out.append(fractions / (fractions.sum(1, keepdims=True) + epsilon))
    return np.concatenate(out)


def fit_rumba(protocol, mask, jobs=1, order=8, n_iter=200, device="cpu"):
    """RUMBA-SD (Canales-Rodriguez et al. 2015) with dipy's kernel and settings; the sphere fODF is returned as SH."""
    from dipy.reconst.rumba import RumbaSDModel, generate_kernel

    gtab = gtab_of(protocol)
    sphere = get_sphere(name="symmetric362")
    voxels = protocol["dwi"].reshape(-1, protocol["dwi"].shape[-1])
    single_fibre = _top_fa(protocol, gtab, protocol["pve"][0])
    tensor_fit = TensorModel(gtab).fit(voxels[single_fibre])
    wm_response = np.array(
        [tensor_fit.evals[:, 0].mean(), tensor_fit.evals[:, 1:].mean(), tensor_fit.evals[:, 1:].mean()]
    )
    model = RumbaSDModel(gtab, wm_response=wm_response, n_iter=n_iter, sphere=sphere, voxelwise=True)
    kernel = generate_kernel(model.gtab, sphere, wm_response, model.gm_response, model.csf_response)
    to_sh = np.linalg.pinv(sh_matrix(sphere.vertices, order))

    def run(indices):
        signals = shm.normalize_data(
            voxels[indices].astype(np.float64), model.where_b0s, min_signal=np.finfo(float).eps
        )
        signals = np.concatenate([np.ones((len(signals), 1)), signals[:, model.where_dwi]], 1)
        signals[signals > 1] = 1
        return rumba_batch(signals, kernel, n_iter, device)[:, :-2] @ to_sh.T

    if str(device).startswith("cuda"):
        chunks = [np.flatnonzero(mask.ravel())]
        parts = [run(chunks[0])]
    else:
        chunks = _chunks(mask, jobs * 4)
        parts = Parallel(n_jobs=jobs)(delayed(run)(indices) for indices in chunks)
    coefficients = np.zeros((mask.size, n_coef(order)), np.float32)
    for indices, part in zip(chunks, parts):
        coefficients[indices] = part
    return np.moveaxis(coefficients.reshape(mask.shape + (n_coef(order),)), -1, 0)


def pad_order(coefficients, order=8):
    """Zero-pad a lower-order SH volume to `order` so every method has the same number of channels."""
    padded = np.zeros((n_coef(order),) + coefficients.shape[1:], np.float32)
    padded[: coefficients.shape[0]] = coefficients
    return padded


def fit_fodf(protocol, mask, method, cfg, jobs=1, device="cpu"):
    """method: msmt_csd (single-shell CSD if the data have one shell), rumba, anything else = csd.
    Returns (n_coef(sh_order_out), X, Y, Z), zero-padded to sh_order_out."""
    multi_shell = len(np.unique(protocol["bvals"][protocol["bvals"] > 0])) > 1
    if method == "msmt_csd" and multi_shell:
        coefficients = fit_msmt(protocol, mask, jobs, device=device)
    elif method == "rumba":
        coefficients = fit_rumba(protocol, mask, jobs, n_iter=cfg.rumba_iters, device=device)
    else:
        coefficients = fit_csd(protocol, mask, jobs)
    return pad_order(coefficients, cfg.sh_order_out)
