"""Train one network.  python -m retinopath.train --variant path --seed 0 [--dev]

One GPU: ordinary training. Several GPUs: start with `torchrun --nproc_per_node=N -m retinopath.train ...`
and the same code runs distributed (one process per GPU). All data sit on the GPU, so there is no loader bottleneck.
"""
import argparse, atexit, math, os, re, signal, socket, time
# patch count and path length change during training; growing segments keep the allocator from fragmenting
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F
from .core import atomic, get_device, load_config, load_json, save_json, seed_all
from .model import PathTracer, build_model, nonneg_loss, path_loss
from . import pipeline as PL

VARIANTS = {                      # name -> changes to cfg.model (and network kind)
    "path": {},                                           # proposed: voxel loss + path-consistency loss
    "voxel": {"lambda_path": 0.0},                        # same network, voxel loss only (the H3 control)
    "onestep": {"one_step": True},                        # ablation: no error accumulation along the path
    "noanat": {"use_anat": False},                        # ablation: no anatomical input channels
    "wmseeds": {"seed_where": "wm"},                      # ablation: path seeds anywhere in white matter
    "mlp": {"lambda_path": 0.0, "kind": "mlp"},           # voxel-wise MLP baseline (no spatial context)
    "fodrefine": {"lambda_path": 0.0, "use_anat": False, "input": "fodf", "kind": "refine"},   # fODF-to-fODF baseline
}


def apply_variant(cfg, variant):
    v = dict(VARIANTS[variant]); kind = v.pop("kind", "unet")
    if "one_step" in v:
        cfg.model.path["one_step"] = v.pop("one_step")
    cfg.model.update(v)
    return kind


def run_dir(cfg, variant, seed):
    d = cfg.P.out / "models" / f"{variant}_s{seed}"; d.mkdir(parents=True, exist_ok=True); return d


# ---------------------------------------------------------------- data on the device
def load_volumes(cfg, sids, protocols, device):
    """Per subject: inputs for every training protocol, reference fODF and anatomy on the device; V1-V3 surface
    seeds stay on the CPU, where patches are drawn."""
    out, P = [], cfg.patch
    for sid in sids:
        ses = PL.session(sid); a = PL.load_anat(cfg, ses)
        pad = [max(0, P - int(s)) for s in a["shape"]]                       # make every axis at least one patch long
        padf = lambda x: F.pad(torch.as_tensor(np.asarray(x, np.float32)), (0, pad[2], 0, pad[1], 0, pad[0]))
        v = np.flatnonzero(a["area"] > 0)
        out.append(dict(
            sid=str(sid), x={p: padf(PL.net_input(cfg, ses, p, a, cfg.model.input)).half().to(device) for p in protocols},
            y=padf(PL.fodf(cfg, ses, "REF", "msmt_csd", anat=a)["sh"]).to(device),
            anat=padf(PL.anat_channels(a)).to(device),
            seeds=torch.as_tensor(a["verts_vox"][v] - cfg.tracking.seed_depth_mm / cfg.ref_voxel * a["normals_vox"][v]).float(),
            inward=torch.as_tensor(-a["normals_vox"][v]).float()))
        out[-1]["wm_idx"] = (out[-1]["anat"][0] > 0.9).nonzero().float().cpu()
    return out


def sample_batch(cfg, vols, protocols, B, gen, device):
    """B random cubic patches (random subject and protocol). 70% are centred near a V1-V3 seed (within +-patch/4),
    the rest anywhere. Returns stacked x, y, anat, path seeds and inward normals (B, S, 3) in patch-voxel units
    (at least 2 voxels from the border), and valid (B, S): all False when the patch holds no usable seed."""
    P, S, m = cfg.patch, cfg.model.path.seeds_per_patch, cfg.model
    X, Y, A, SD, IN, VA = [], [], [], [], [], []
    for _ in range(B):                                   # positions are drawn on the CPU; only the patches are cut on the device
        v = vols[int(torch.randint(len(vols), (1,), generator=gen))]
        p = protocols[int(torch.randint(len(protocols), (1,), generator=gen))]
        size = torch.tensor(v["y"].shape[1:])
        if torch.rand(1, generator=gen) < 0.7 and len(v["seeds"]):               # most patches sit on visual cortex
            c = v["seeds"][int(torch.randint(len(v["seeds"]), (1,), generator=gen))] + (torch.rand(3, generator=gen) - 0.5) * P / 2
        else:
            c = torch.rand(3, generator=gen) * size
        o = torch.minimum(torch.clamp((c - P / 2).round().long(), min=0), size - P)
        sl = (slice(None),) + tuple(slice(i, i + P) for i in o.tolist())
        X.append(v["x"][p][sl].float()); Y.append(v["y"][sl]); A.append(v["anat"][sl])
        pts = (v["wm_idx"] if m.seed_where == "wm" else v["seeds"]) - o
        ok = ((pts > 2) & (pts < P - 3)).all(1).nonzero()[:, 0]
        if len(ok):
            pick = ok[torch.randint(len(ok), (S,), generator=gen)]
            inw = F.normalize(torch.randn(S, 3, generator=gen), dim=-1) if m.seed_where == "wm" else v["inward"][pick]   # wm: ablation
            SD.append(pts[pick]); IN.append(inw); VA.append(torch.ones(S, dtype=torch.bool))
        else:
            SD.append(torch.full((S, 3), P / 2.0)); IN.append(torch.zeros(S, 3)); VA.append(torch.zeros(S, dtype=torch.bool))
    return [torch.stack(t) for t in (X, Y, A)] + [torch.stack(t).to(device) for t in (SD, IN, VA)]


def net_in(cfg, x, anat):
    return torch.cat([x, anat], 1) if cfg.model.use_anat else x


# ---------------------------------------------------------------- loss
def losses(cfg, model, tracer, batch, steps, scale, amp_dev):
    """Returns (total loss, {part: float}). Voxel MSE (inside wm > wm_mask_thr) and the non-negativity penalty are
    divided by scale**2 and scale (std of the training fODF coefficients); the path loss (mm) is added as a
    fraction of the path length steps * step_mm."""
    x, y, anat, seeds, inward, valid = batch
    m = cfg.model
    with torch.autocast(amp_dev, enabled=bool(m.amp) and amp_dev == "cuda"):
        pred = model(net_in(cfg, x, anat))
    pred = pred.float()
    wm = anat[:, :1]; mask = (wm > cfg.wm_mask_thr).float()
    out = {"voxel": ((pred - y) ** 2 * mask).sum() / (mask.sum() * y.shape[1] + 1e-8) / scale ** 2,
           "nonneg": nonneg_loss(tracer, pred, wm) / scale}
    total = out["voxel"] + m.lambda_nonneg * out["nonneg"]
    if m.lambda_path > 0:
        sv = m.path.step_mm / cfg.ref_voxel
        out["path_mm"] = path_loss(tracer, pred, y, wm, seeds, inward, valid, steps, sv, cfg.ref_voxel, m.path.one_step)
        total = total + m.lambda_path * out["path_mm"] / (steps * m.path.step_mm)     # error as a fraction of path length
    return total, {k: float(v.detach()) for k, v in out.items()}


@torch.no_grad()
def predict(cfg, model, x, anat, device):
    """Whole-crop inference. x: (C, X, Y, Z) numpy network input; anat: 7-channel numpy. Returns (45, X, Y, Z) float32."""
    model.eval()
    t = torch.as_tensor(np.concatenate([x.astype(np.float32), anat]) if cfg.model.use_anat else x.astype(np.float32))[None].to(device)
    k = 2 ** (cfg.model.levels - 1); pad = [(-s) % k for s in t.shape[2:]]
    t = F.pad(t, (0, pad[2], 0, pad[1], 0, pad[0]))
    with torch.autocast(device.type, enabled=bool(cfg.model.amp) and device.type == "cuda"):
        y = model(t)
    s = x.shape[1:]
    return y[0, :, :s[0], :s[1], :s[2]].float().cpu().numpy()


def auto_batch(cfg, model, tracer, vols, protocols, device, scale, resumed=None):
    """Largest batch (8, 16, 32, 64) whose training step fits in 70% of GPU memory; 4 if even 8 does not.
    A size is tried only if doubling the memory of the previous one stays under that limit, because a machine
    whose GPU shares memory with the system can stall rather than report that it is full. Not on CUDA: 2. An integer batch_size in config.yaml is used as given. A resumed run keeps the batch it
    started with (`resumed`), so the search runs once per training run. The trial step always includes the path
    loss, so networks that differ only in the loss end up with the same batch."""
    if cfg.batch_size != "auto":
        return int(cfg.batch_size)
    if resumed:
        return int(resumed)
    if device.type != "cuda":
        return 2
    best, gen = 4, torch.Generator().manual_seed(0)
    lam = cfg.model.lambda_path; cfg.model["lambda_path"] = 1.0
    held = peak = torch.cuda.memory_allocated(device)                # volumes and weights
    limit = 0.7 * torch.cuda.mem_get_info(device)[1]
    for b in (8, 16, 32, 64):
        if held + 2 * (peak - held) > limit:
            break
        try:
            torch.cuda.reset_peak_memory_stats(device)
            total, _ = losses(cfg, model, tracer, sample_batch(cfg, vols, protocols, b, gen, device), cfg.model.path.steps_end, scale, "cuda")
            total.backward(); model.zero_grad(set_to_none=True); del total
            peak = torch.cuda.max_memory_allocated(device)
            if peak > limit:
                break
            best = b
        except torch.OutOfMemoryError:
            break
    cfg.model["lambda_path"] = lam; torch.cuda.empty_cache()
    return best


# ---------------------------------------------------------------- training loop
def train(cfg, variant="path", seed=0, subjects=None):
    """Train one (variant, seed) into outputs/models/<variant>_s<seed>/ (last.pt, log.json); resumes from last.pt.
    Progress = max(step / (epochs * steps_per_epoch), hours / max_train_hours); it drives the cosine learning rate
    and the path-length curriculum, and training ends at 1. SIGINT/SIGTERM save a checkpoint and exit with 130."""
    kind = apply_variant(cfg, variant)
    ddp = "LOCAL_RANK" in os.environ
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if ddp else (0, 1)
    if ddp:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"])) if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        torch.distributed.init_process_group("nccl" if device.type == "cuda" else "gloo")
    else:
        device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    seed_all(cfg.seed + 1000 * seed + rank)
    out = run_dir(cfg, variant, seed); ck = out / "last.pt"
    take_over(out / "pid", owner=rank == 0)
    prev = torch.load(ck, map_location="cpu", weights_only=False) if ck.exists() else None
    total_steps = cfg.epochs * cfg.steps_per_epoch; cap = cfg.max_train_hours
    if prev and prev["meta"].get("steps", total_steps) != total_steps:           # every network trains equally long
        raise SystemExit(f"{out.name} was started with {prev['meta']['steps']} training steps and config.yaml now gives "
                         f"{total_steps}; delete {out} to train it again")
    if prev and prev["state"]["done"]:
        return out                                               # this run already finished
    if prev is None and rank == 0:
        clear_outputs(cfg, variant, seed)
    subjects = subjects or load_json(cfg.P.out / "subjects.json")
    protos = list(cfg.train_protocols)
    tr = load_volumes(cfg, subjects["train"], protos, device)
    va = load_volumes(cfg, subjects["val"], protos, device)
    scale = float(torch.cat([v["y"][:, v["anat"][0] > 0.5].flatten() for v in tr]).std())      # fODF coefficient scale

    model = build_model(cfg, kind).to(device)
    if cfg.model.compile:
        model = torch.compile(model)
    tracer = PathTracer(cfg.sh_order_out, cfg.model.path.cone_deg, cfg.model.path.sharpen).to(device)
    B = auto_batch(cfg, model, tracer, tr, protos, device, scale, prev["meta"]["batch"] if prev else None)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None) if ddp else model
    lr0 = cfg.model.lr * world
    opt = torch.optim.AdamW(net.parameters(), lr=lr0, weight_decay=cfg.model.weight_decay)
    scaler = torch.amp.GradScaler(enabled=bool(cfg.model.amp) and device.type == "cuda")
    state = dict(step=0, hours=0.0, log=[], done=False)
    if prev:                                                                    # resume
        model.load_state_dict(prev["model"]); opt.load_state_dict(prev["opt"]); scaler.load_state_dict(prev["scaler"])
        state = prev["state"]
    gen = torch.Generator().manual_seed(cfg.seed + 1000 * seed + rank + state["step"])
    vgen = lambda: torch.Generator().manual_seed(123)
    t0 = last_ck = time.time(); h0 = state["hours"]; pc = cfg.model.path

    def save(done=False):
        state["hours"], state["done"] = h0 + (time.time() - t0) / 3600, done
        atomic(ck, lambda f: torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), state=state,
                                             meta=dict(variant=variant, seed=seed, kind=kind, batch=B, world=world, scale=scale, steps=total_steps)), f))
        save_json(state["log"], out / "log.json")

    stop = {"now": False}                                        # Ctrl-C / kernel interrupt / kill: save first, then leave
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(now=True))
    if rank == 0 and state["step"]:
        print(f"resuming {variant}_s{seed} at step {state['step']} ({state['hours']:.2f} h done, batch {B})", flush=True)

    parent = os.getppid()
    while state["step"] < total_steps:
        hours = h0 + (time.time() - t0) / 3600
        if os.getppid() != parent:                               # the notebook or shell that started this run is gone
            stop["now"] = True
        if ddp:                                                  # rank 0 decides, so all ranks stop at the same step
            ctl = torch.tensor([float(stop["now"]), hours], device=device)
            torch.distributed.broadcast(ctl, 0); stop["now"], hours = bool(ctl[0]), float(ctl[1])
        if stop["now"]:
            if rank == 0:
                save(); print(f"stopped at step {state['step']}; checkpoint saved, run again to continue", flush=True)
            raise SystemExit(130)
        prog = max(state["step"] / total_steps, hours / cap if cap else 0.0)
        if prog >= 1:
            break
        steps = int(round(pc.steps_start + (pc.steps_end - pc.steps_start) * min(1.0, 2 * prog)))     # path-length curriculum
        for g in opt.param_groups:
            g["lr"] = lr0 * 0.5 * (1 + math.cos(math.pi * prog))
        net.train()
        total, parts = losses(cfg, net, tracer, sample_batch(cfg, tr, protos, B, gen, device), steps, scale, device.type)
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update(); state["step"] += 1
        if rank == 0 and state["step"] % cfg.steps_per_epoch == 0:
            net.eval(); vl = []
            with torch.no_grad():
                # every variant reports the same validation numbers: voxel loss and accumulated path error
                g2 = vgen(); old = cfg.model.lambda_path, pc.one_step; cfg.model["lambda_path"], pc["one_step"] = 1.0, False
                for _ in range(8):
                    vl.append(losses(cfg, model, tracer, sample_batch(cfg, va, protos, B, g2, device), pc.steps_end, scale, device.type)[1])
                cfg.model["lambda_path"], pc["one_step"] = old
            row = dict(step=state["step"], hours=round(hours, 4), hours_left=round(min(hours / state["step"] * (total_steps - state["step"]), cap - hours if cap else math.inf), 2),
                       path_steps=steps, **{f"train_{k}": v for k, v in parts.items()},
                       **{f"val_{k}": float(np.mean([r[k] for r in vl])) for k in vl[0]})
            if device.type == "cuda":                            # peak since the last row; reserved far above used = fragmentation
                row["gpu_gb"], row["gpu_reserved_gb"] = (round(x / 2 ** 30, 1) for x in (torch.cuda.max_memory_allocated(device), torch.cuda.memory_reserved(device)))
                torch.cuda.reset_peak_memory_stats(device)
            state["log"].append(row); print(row, flush=True)
        if rank == 0 and time.time() - last_ck > cfg.model.ckpt_minutes * 60:
            save(); last_ck = time.time()
    if rank == 0:
        save(done=True)
    if ddp:
        torch.distributed.destroy_process_group()
    return out


def take_over(lock, owner=True, wait=600):
    """One process per run folder. `lock` holds "host pid" of the process training there. If that process is still
    alive on this machine (a run left behind by a killed notebook), it is asked to save and stop, and this call
    waits for it. Processes of the same torchrun group are not each other's rivals. owner: write the lock."""
    host, me = socket.gethostname(), os.getpid()
    if lock.exists():
        h, _, pid = lock.read_text().partition(" ")
        pid = int(pid) if pid.strip().isdigit() else -1
        alive = lambda: _alive(pid)
        if h == host and pid not in (-1, me) and alive() and os.getpgid(pid) != os.getpgid(0):
            if owner:
                print(f"another process ({pid}) is training this run; asking it to save and stop", flush=True)
                os.kill(pid, signal.SIGINT)
            t0 = time.time()
            while alive() and time.time() - t0 < wait:
                time.sleep(1)
            if alive():
                raise RuntimeError(f"process {pid} still holds {lock.parent}; stop it and run again")
    if owner:
        lock.write_text(f"{host} {me}")
        atexit.register(lambda: lock.exists() and lock.read_text() == f"{host} {me}" and lock.unlink())


def _alive(pid):
    try:
        os.kill(pid, 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def clear_outputs(cfg, variant, seed):
    """Remove cached fODFs, tractograms and scores of an earlier network with this name, so that a retrained
    network is never mixed with results of the old one."""
    pat = re.compile(rf"_net_{re.escape(variant)}_s{seed}([._+]|$)")
    for f in cfg.P.cache.glob(f"*/*_net_{variant}_s{seed}*"):
        if pat.search(f.name):
            f.unlink()


def load_trained(cfg, variant, seed, device):
    """Returns (model, cfg adjusted to the variant)."""
    kind = apply_variant(cfg, variant)
    s = torch.load(run_dir(cfg, variant, seed) / "last.pt", map_location=device, weights_only=False)
    model = build_model(cfg, kind).to(device); model.load_state_dict(s["model"]); model.eval()
    return model, s["meta"]


def net_fodf(cfg, ses, name, method, model, device):
    """Run a trained network on protocol `name` and cache its fODF under `method` (e.g. net_path_s0)."""
    f = PL.sdir(cfg, ses) / f"fodf_{name}_{method}.npz"
    if f.exists():
        return
    a = PL.load_anat(cfg, ses)
    sh = predict(cfg, model, PL.net_input(cfg, ses, name, a, cfg.model.input), PL.anat_channels(a), device)
    sh = sh * (a["pve"][0] > cfg.wm_mask_thr)
    PL.save_fodf(cfg, ses, name, method, sh, a["affine"], a["pve"])


def trained_runs(cfg):
    """[(variant, seed)] of the runs listed in config.yaml that have finished training."""
    from .jobs import train_commands
    out = []
    for name, _ in train_commands(cfg):
        ck = cfg.P.out / "models" / name / "last.pt"
        if ck.exists() and torch.load(ck, map_location="cpu", weights_only=False)["state"]["done"]:
            v, s = name.rsplit("_s", 1); out.append((v, int(s)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="path", choices=list(VARIANTS)); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    train(load_config(a.dev), a.variant, a.seed)
