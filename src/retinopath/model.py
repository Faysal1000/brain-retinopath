"""Network, differentiable short-streamline propagator and the path-consistency loss."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dipy.data import get_sphere
from .recon import n_coef, sh_matrix

N_ANAT = 7  # wm, gm, csf, signed distance, inward normal (3)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.a = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8 if out_channels % 8 == 0 else 4, out_channels),
            nn.SiLU(),
        )
        self.b = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8 if out_channels % 8 == 0 else 4, out_channels),
            nn.SiLU(),
        )
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.b(self.a(x)) + self.skip(x)


class UNet3D(nn.Module):
    """Residual 3D U-Net: base * 2**i channels at level i, average pooling down, trilinear upsampling and skips up."""

    def __init__(self, in_channels, out_channels, base=48, levels=3):
        super().__init__()
        widths = [base * 2**level for level in range(levels)]
        self.down = nn.ModuleList(
            [Block(in_channels if level == 0 else widths[level - 1], widths[level]) for level in range(levels)]
        )
        self.up = nn.ModuleList(
            [Block(widths[level] + widths[level + 1], widths[level]) for level in range(levels - 1)]
        )
        self.head = nn.Conv3d(widths[0], out_channels, 1)
        self.levels = levels

    def forward(self, x):
        skips = []
        for level, block in enumerate(self.down):
            x = block(x if level == 0 else F.avg_pool3d(x, 2))
            skips.append(x)
        for level in reversed(range(self.levels - 1)):
            x = F.interpolate(x, size=skips[level].shape[2:], mode="trilinear", align_corners=False)
            x = self.up[level](torch.cat([skips[level], x], 1))
        return self.head(x)


class VoxelMLP(nn.Module):
    """Voxel-wise baseline in the spirit of Karimi et al. 2021: no spatial context (1x1x1 convolutions)."""

    def __init__(self, in_channels, out_channels, width=256, depth=4):
        super().__init__()
        layers, channels = [], in_channels
        for _ in range(depth):
            layers += [nn.Conv3d(channels, width, 1), nn.SiLU()]
            channels = width
        self.net = nn.Sequential(*layers, nn.Conv3d(channels, out_channels, 1))

    def forward(self, x):
        return self.net(x)


class FODRefine(nn.Module):
    """Input fODF (first n_coefficients channels, then anatomy) plus a U-Net correction (as in FOD-Net, Zeng et al.
    2022). The head starts at zero, so training starts from the input fODF."""

    def __init__(self, n_coefficients, base=48, levels=3, in_channels=None):
        super().__init__()
        self.n_coefficients = n_coefficients
        self.net = UNet3D(in_channels or n_coefficients, n_coefficients, base, levels)
        nn.init.zeros_(self.net.head.weight)
        nn.init.zeros_(self.net.head.bias)

    def forward(self, x):
        return x[:, : self.n_coefficients] + self.net(x)


VARIANTS = {  # name -> changes to cfg.model; "kind" picks the network (default cfg.model.kind)
    "path": {},  # proposed: voxel loss + path-consistency loss
    "voxel": {"lambda_path": 0.0},  # same network, voxel loss only (the H3 control)
    "onestep": {"one_step": True},  # ablation: no error accumulation along the path
    "noanat": {"use_anat": False},  # ablation: no anatomical input channels
    "wmseeds": {"seed_where": "wm"},  # ablation: path seeds anywhere in white matter
    "signal": {"lambda_path": 0.0, "input": "signal", "kind": "unet"},  # baseline: U-Net from the diffusion signal
    "mlp": {"lambda_path": 0.0, "input": "signal", "kind": "mlp"},  # baseline: voxel-wise MLP, no spatial context
}


def apply_variant(cfg, variant):
    """Applies the variant's changes to cfg.model in place; returns the network kind."""
    changes = dict(VARIANTS[variant])
    kind = changes.pop("kind", cfg.model.kind)
    if "one_step" in changes:
        cfg.model.path["one_step"] = changes.pop("one_step")
    cfg.model.update(changes)
    return kind


def input_sources(cfg):
    """Network inputs ('signal', 'fodf') that the configured variants read."""
    return {VARIANTS[variant].get("input", cfg.model.input) for variant in cfg.variants}


def build_model(cfg, kind="unet"):
    """kind: refine | unet | mlp. Input channels: n_coef(sh_order_out) for fODF input (refine) or
    len(shells_in) * n_coef(sh_order_in) for signal input, plus N_ANAT if use_anat; output: n_coef(sh_order_out)."""
    out_channels = n_coef(cfg.sh_order_out)
    anat_channels = N_ANAT if cfg.model.use_anat else 0
    if kind == "refine":
        return FODRefine(out_channels, cfg.base_channels, cfg.model.levels, out_channels + anat_channels)
    in_channels = len(cfg.shells_in) * n_coef(cfg.sh_order_in) + anat_channels
    return (
        VoxelMLP(in_channels, out_channels)
        if kind == "mlp"
        else UNet3D(in_channels, out_channels, cfg.base_channels, cfg.model.levels)
    )


# ------------------------------------------------------------------------------------------------ propagator
class _SparseGrad(torch.autograd.Function):
    """Identity; makes the sparse gradient of F.embedding dense."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return gradient.to_dense() if gradient.is_sparse else gradient


def voxel_table(volume):
    """(B, C, X, Y, Z) -> (B * X * Y * Z, C) lookup table for sample_trilinear; sparse gradients, made dense once."""
    return _SparseGrad.apply(volume.permute(0, 2, 3, 4, 1).reshape(-1, volume.shape[1]))


def sample_trilinear(volume, points, table=None):
    """volume: (B, C, X, Y, Z); points: (B, S, 3) voxel indices, clamped to the border. Returns (B, S, C).
    table: voxel_table(volume), for repeated sampling of the same volume."""
    batch, channels, size_x, size_y, size_z = volume.shape
    table = voxel_table(volume) if table is None else table
    size = torch.tensor([size_x, size_y, size_z], device=volume.device, dtype=points.dtype)
    clamped = torch.minimum(torch.clamp(points, min=0), size - 1.001)
    corner = clamped.floor().long()
    fraction = clamped - corner
    offsets = torch.tensor(
        [[dx, dy, dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)], device=volume.device
    )  # (8, 3)
    neighbours = corner[:, None] + offsets[None, :, None]  # (B, 8, S, 3)
    rows = (neighbours[..., 0] * size_y + neighbours[..., 1]) * size_z + neighbours[..., 2]  # (B, 8, S)
    rows = (
        rows + (torch.arange(batch, device=volume.device) * (size_x * size_y * size_z))[:, None, None]
    )  # rows of the table
    weights = torch.where(offsets[None, :, None].bool(), fraction[:, None], 1 - fraction[:, None])
    weights = weights[..., 0] * weights[..., 1] * weights[..., 2]  # (B, 8, S)
    values = F.embedding(rows, table, sparse=True)  # (B, 8, S, C)
    out = 0
    for neighbour in range(8):
        out = out + values[:, neighbour] * weights[:, neighbour, :, None]
    return out


class PathTracer(nn.Module):
    """Differentiable deterministic tracker. Each step goes along the mean of the symmetric362 directions within
    cone_deg of the previous one, weighted by (amplitude / max amplitude) ** sharpen. Voxel coordinates."""

    def __init__(self, order=8, cone_deg=45.0, sharpen=4.0):
        super().__init__()
        directions = get_sphere(name="symmetric362").vertices.astype(np.float32)
        self.register_buffer("directions", torch.from_numpy(directions))
        self.register_buffer("basis", torch.from_numpy(sh_matrix(directions, order)))
        self.cone_cos, self.sharpen = float(np.cos(np.radians(cone_deg))), sharpen

    def amplitudes(self, volume, points, table=None):
        """fODF amplitudes (B, S, 362) on the sphere directions at `points`, clipped at 0."""
        return torch.relu(sample_trilinear(volume, points, table) @ self.basis.T)  # (B, S, K)

    def start_directions(self, volume, seeds, inward):
        """Hard start direction: the sphere direction maximising amplitude * relu(cos to inward normal) ** 2."""
        score = self.amplitudes(volume, seeds) * torch.relu(inward @ self.directions.T) ** 2
        return self.directions[score.argmax(-1)]

    def step_direction(self, volume, points, previous, table=None):
        amplitudes = self.amplitudes(volume, points, table)
        weights = (amplitudes / (amplitudes.amax(-1, keepdim=True) + 1e-8)) ** self.sharpen * (
            (previous @ self.directions.T) > self.cone_cos
        )
        direction = weights @ self.directions
        norm = direction.norm(dim=-1, keepdim=True)
        return torch.where(norm > 1e-4, direction / (norm + 1e-8), previous)

    def trace(self, volume, seeds, start, steps, step_vox):
        """Euler steps of length step_vox (voxels) from seeds with start directions `start`. Returns positions (B, S, steps + 1, 3)."""
        points, direction, path, table = seeds, start, [seeds], voxel_table(volume)
        for _ in range(steps):
            direction = self.step_direction(volume, points, direction, table)
            points = points + step_vox * direction
            path.append(points)
        return torch.stack(path, 2)


def path_loss(tracer, pred, ref, wm, seeds, inward, valid, steps, step_vox, voxel_mm, one_step=False):
    """Path-consistency loss (mm) between paths traced on pred and on ref from the same seeds and start directions.
    pred/ref: (B, 45, X, Y, Z); wm: (B, 1, X, Y, Z); seeds, inward: (B, S, 3) patch voxels; valid: (B, S).
    Points count until the reference path leaves the patch interior or white matter (wm > 0.1). Huber distance
    (delta 2 voxels) times voxel_mm; one_step: 1 - cos of the step directions at the reference positions."""
    size = torch.tensor(pred.shape[2:], device=pred.device, dtype=pred.dtype)
    with torch.no_grad():
        start = tracer.start_directions(ref, seeds, inward)
        ref_path = tracer.trace(ref, seeds, start, steps, step_vox)  # (B, S, T+1, 3)
        batch, n_seeds, n_points, _ = ref_path.shape
        inside = ((ref_path > 1) & (ref_path < size - 2)).all(-1)
        in_wm = sample_trilinear(wm, ref_path.reshape(batch, -1, 3)).reshape(batch, n_seeds, n_points) > 0.1
        alive = torch.cumprod((inside & in_wm).float(), dim=2) * valid[:, :, None]  # stays 0 after leaving
    if alive[:, :, 1:].sum() < 1:
        return pred.sum() * 0
    if one_step:  # ablation: direction error at reference positions, errors do not accumulate
        ref_steps = F.normalize(ref_path[:, :, 1:] - ref_path[:, :, :-1], dim=-1)
        previous = torch.cat([start[:, :, None], ref_steps[:, :, :-1]], 2).reshape(batch, -1, 3)
        pred_steps = tracer.step_direction(pred, ref_path[:, :, :-1].reshape(batch, -1, 3), previous).reshape(
            batch, n_seeds, n_points - 1, 3
        )
        error = (1 - (pred_steps * ref_steps).sum(-1)) * step_vox * voxel_mm * steps / 2  # about the mm scale
    else:
        pred_path = tracer.trace(pred, seeds, start, steps, step_vox)
        error = F.huber_loss(pred_path[:, :, 1:], ref_path[:, :, 1:], reduction="none", delta=2.0).sum(-1) * voxel_mm
    counted = alive[:, :, 1:]
    return (error * counted).sum() / counted.sum()


@torch.no_grad()
def path_error(tracer, pred, ref, wm, seeds, inward, steps, step_vox, voxel_mm):
    """Distance (mm) between paths traced on pred and on ref from the same seeds and start directions: mean over
    the points before the reference path leaves white matter, and at the last of those points. Volumes (1, C, X, Y, Z),
    seeds and inward (1, S, 3) in voxels. Returns (mean, end, number of paths counted)."""
    size = torch.tensor(pred.shape[2:], device=pred.device, dtype=pred.dtype)
    start = tracer.start_directions(ref, seeds, inward)
    ref_path = tracer.trace(ref, seeds, start, steps, step_vox)
    inside = ((ref_path > 1) & (ref_path < size - 2)).all(-1)
    in_wm = sample_trilinear(wm, ref_path.reshape(1, -1, 3)).reshape(inside.shape) > 0.1
    counted = torch.cumprod((inside & in_wm).float(), dim=2)[:, :, 1:]
    distance = (tracer.trace(pred, seeds, start, steps, step_vox) - ref_path)[:, :, 1:].norm(dim=-1) * voxel_mm
    n_points = counted.sum(2)
    if n_points.sum() < 1:
        return float("nan"), float("nan"), 0
    last = torch.gather(distance, 2, (n_points.long() - 1).clamp(min=0)[..., None])[..., 0]
    return (
        float((distance * counted).sum() / counted.sum()),
        float(last[n_points > 0].mean()),
        int((n_points > 0).sum()),
    )


def nonneg_loss(tracer, pred, wm, n_voxels=2000):
    """Penalty on negative fODF amplitudes at random white-matter voxels."""
    batch, channels = pred.shape[:2]
    voxels, wm_weight = pred.reshape(batch, channels, -1), wm.reshape(batch, -1)
    picked = torch.randint(0, voxels.shape[-1], (batch, n_voxels), device=pred.device)
    amplitudes = torch.gather(voxels, 2, picked[:, None].expand(-1, channels, -1)).transpose(1, 2) @ tracer.basis.T
    return (torch.relu(-amplitudes) * torch.gather(wm_weight, 1, picked)[..., None]).mean()
