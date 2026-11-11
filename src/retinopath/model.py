"""Network, differentiable short-streamline propagator and the path-consistency loss."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dipy.data import get_sphere
from .recon import n_coef, sh_matrix

N_ANAT = 7        # wm, gm, csf, signed distance, inward normal (3)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.a = nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(8 if cout % 8 == 0 else 4, cout), nn.SiLU())
        self.b = nn.Sequential(nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(8 if cout % 8 == 0 else 4, cout), nn.SiLU())
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.b(self.a(x)) + self.skip(x)


class UNet3D(nn.Module):
    """Plain residual 3D U-Net (kept standard so that only the loss differs between variants). Level i has
    base * 2**i channels; 2x average pooling down, trilinear upsampling and concatenated skips up."""

    def __init__(self, cin, cout, base=48, levels=3):
        super().__init__()
        ch = [base * 2 ** i for i in range(levels)]
        self.down = nn.ModuleList([Block(cin if i == 0 else ch[i - 1], ch[i]) for i in range(levels)])
        self.up = nn.ModuleList([Block(ch[i] + ch[i + 1], ch[i]) for i in range(levels - 1)])
        self.head = nn.Conv3d(ch[0], cout, 1)
        self.levels = levels

    def forward(self, x):
        feats = []
        for i, blk in enumerate(self.down):
            x = blk(x if i == 0 else F.avg_pool3d(x, 2)); feats.append(x)
        for i in reversed(range(self.levels - 1)):
            x = F.interpolate(x, size=feats[i].shape[2:], mode="trilinear", align_corners=False)
            x = self.up[i](torch.cat([feats[i], x], 1))
        return self.head(x)


class VoxelMLP(nn.Module):
    """Voxel-wise baseline in the spirit of Karimi et al. 2021: no spatial context (1x1x1 convolutions)."""

    def __init__(self, cin, cout, width=256, depth=4):
        super().__init__()
        layers, c = [], cin
        for _ in range(depth):
            layers += [nn.Conv3d(c, width, 1), nn.SiLU()]; c = width
        self.net = nn.Sequential(*layers, nn.Conv3d(c, cout, 1))

    def forward(self, x):
        return self.net(x)


class FODRefine(nn.Module):
    """fODF-to-fODF enhancement in the spirit of FOD-Net (Zeng et al. 2022): the input is the classical fODF of the
    degraded scan and the network predicts the residual to the reference fODF. Same U-Net as the other variants, so
    the comparison isolates the input (fODF instead of signal) from the architecture."""

    def __init__(self, n, base=48, levels=3):
        super().__init__()
        self.n, self.net = n, UNet3D(n, n, base, levels)
        nn.init.zeros_(self.net.head.weight); nn.init.zeros_(self.net.head.bias)      # start as the identity

    def forward(self, x):
        return x[:, :self.n] + self.net(x)


def build_model(cfg, kind="unet"):
    """kind: unet | mlp | refine. Input channels: len(shells_in) * n_coef(sh_order_in) for signal input or
    n_coef(sh_order_out) for fODF input, plus N_ANAT if use_anat; output channels: n_coef(sh_order_out)."""
    cout = n_coef(cfg.sh_order_out)
    if kind == "refine":
        return FODRefine(cout, cfg.base_channels, cfg.model.levels)
    cin = len(cfg.shells_in) * n_coef(cfg.sh_order_in) + (N_ANAT if cfg.model.use_anat else 0)
    return VoxelMLP(cin, cout) if kind == "mlp" else UNet3D(cin, cout, cfg.base_channels, cfg.model.levels)


# ------------------------------------------------------------------------------------------------ propagator
def sample_trilinear(vol, pts):
    """vol: (B, C, X, Y, Z); pts: (B, S, 3) in voxel index units. Returns (B, S, C). Differentiable in vol and pts.
    Points outside the volume are clamped to the border. Written with gather so it runs on CUDA, MPS and CPU."""
    B, C, X, Y, Z = vol.shape
    size = torch.tensor([X, Y, Z], device=vol.device, dtype=pts.dtype)
    p = torch.minimum(torch.clamp(pts, min=0), size - 1.001)
    i0 = p.floor().long(); f = p - i0
    flat = vol.reshape(B, C, -1)
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                idx = ((i0[..., 0] + dx) * Y + (i0[..., 1] + dy)) * Z + (i0[..., 2] + dz)          # (B, S)
                w = (f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1]) * (f[..., 2] if dz else 1 - f[..., 2])
                out = out + torch.gather(flat, 2, idx[:, None].expand(-1, C, -1)) * w[:, None]
    return out.transpose(1, 2)


class PathTracer(nn.Module):
    """Soft deterministic tracking: at each step the new direction is the amplitude-weighted mean of the fODF
    inside a cone around the previous direction (a soft arg-max), so positions are differentiable in the fODF.

    fODF amplitudes are evaluated on the 362 directions of symmetric362 and clipped at 0. Step weights are
    (amplitude / max amplitude) ** sharpen, zeroed outside `cone_deg` of the previous direction; if all weights
    vanish the previous direction is kept. Coordinates are voxel units of the volume passed in (patch frame)."""

    def __init__(self, order=8, cone_deg=45.0, sharpen=4.0):
        super().__init__()
        u = get_sphere(name="symmetric362").vertices.astype(np.float32)
        self.register_buffer("U", torch.from_numpy(u))
        self.register_buffer("Bm", torch.from_numpy(sh_matrix(u, order)))
        self.cos, self.p = float(np.cos(np.radians(cone_deg))), sharpen

    def amps(self, vol, pts):
        return torch.relu(sample_trilinear(vol, pts) @ self.Bm.T)                                   # (B, S, K)

    def start_dirs(self, vol, seeds, inward):
        """Hard start direction: the sphere direction maximising amplitude * relu(cos to inward normal) ** 2."""
        w = self.amps(vol, seeds) * torch.relu(inward @ self.U.T) ** 2
        return self.U[w.argmax(-1)]

    def step_dir(self, vol, pts, prev):
        a = self.amps(vol, pts)
        w = (a / (a.amax(-1, keepdim=True) + 1e-8)) ** self.p * ((prev @ self.U.T) > self.cos)
        d = w @ self.U
        n = d.norm(dim=-1, keepdim=True)
        return torch.where(n > 1e-4, d / (n + 1e-8), prev)

    def trace(self, vol, seeds, d0, steps, step_vox):
        """Euler steps of length step_vox (voxels) from seeds with start directions d0. Returns positions (B, S, steps + 1, 3)."""
        pts, d, out = seeds, d0, [seeds]
        for _ in range(steps):
            d = self.step_dir(vol, pts, d)
            pts = pts + step_vox * d
            out.append(pts)
        return torch.stack(out, 2)


def path_loss(tracer, pred, ref, wm, seeds, inward, valid, steps, step_vox, voxel_mm, one_step=False):
    """Path-consistency loss (mm). pred/ref: (B, 45, X, Y, Z) fODF; wm: (B, 1, X, Y, Z); seeds, inward: (B, S, 3)
    in patch-voxel units; valid: (B, S) bool. Reference paths carry no gradient.

    Both paths start from the same seeds and the same start directions (taken from `ref`). A reference point counts
    while the path is more than 1 voxel inside the patch and in white matter (wm > 0.1); once it leaves, all later
    points are masked. Default: Huber distance (delta 2 voxels per axis, summed over axes) between predicted and
    reference positions, times voxel_mm. one_step: 1 - cos between predicted and reference step directions at the
    reference positions, scaled by path length / 2. Mean over counted points; 0 if there are none."""
    X = torch.tensor(pred.shape[2:], device=pred.device, dtype=pred.dtype)
    with torch.no_grad():
        d0 = tracer.start_dirs(ref, seeds, inward)
        pr = tracer.trace(ref, seeds, d0, steps, step_vox)                                           # (B, S, T+1, 3)
        Bn, S, T1, _ = pr.shape
        inside = ((pr > 1) & (pr < X - 2)).all(-1)
        in_wm = sample_trilinear(wm, pr.reshape(Bn, -1, 3)).reshape(Bn, S, T1) > 0.1
        alive = torch.cumprod((inside & in_wm).float(), dim=2) * valid[:, :, None]                    # stays 0 after leaving
    if alive[:, :, 1:].sum() < 1:
        return pred.sum() * 0
    if one_step:        # ablation: direction error at reference positions, errors do not accumulate
        dref = F.normalize(pr[:, :, 1:] - pr[:, :, :-1], dim=-1)
        prev = torch.cat([d0[:, :, None], dref[:, :, :-1]], 2).reshape(Bn, -1, 3)
        dp = tracer.step_dir(pred, pr[:, :, :-1].reshape(Bn, -1, 3), prev).reshape(Bn, S, T1 - 1, 3)
        err = (1 - (dp * dref).sum(-1)) * step_vox * voxel_mm * steps / 2                             # roughly the mm scale of the default loss
    else:
        pp = tracer.trace(pred, seeds, d0, steps, step_vox)
        err = F.huber_loss(pp[:, :, 1:], pr[:, :, 1:], reduction="none", delta=2.0).sum(-1) * voxel_mm
    m = alive[:, :, 1:]
    return (err * m).sum() / m.sum()


def nonneg_loss(tracer, pred, wm, n=2000):
    """Penalty on negative fODF amplitudes at random white-matter voxels."""
    Bn, C = pred.shape[:2]
    flat, w = pred.reshape(Bn, C, -1), wm.reshape(Bn, -1)
    idx = torch.randint(0, flat.shape[-1], (Bn, n), device=pred.device)
    a = torch.gather(flat, 2, idx[:, None].expand(-1, C, -1)).transpose(1, 2) @ tracer.Bm.T
    return (torch.relu(-a) * torch.gather(w, 1, idx)[..., None]).mean()
