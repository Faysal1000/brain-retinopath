"""Train one network: python -m retinopath.train --variant path --seed 0 [--dev] (or under torchrun for DDP)."""

import argparse, atexit, copy, json, math, os, random, re, signal, socket, time

# patch count and path length change during training; growing segments keep the allocator from fragmenting
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F
from .core import atomic, get_device, load_config, load_json, save_json, seed_all
from .model import VARIANTS, PathTracer, apply_variant, build_model, nonneg_loss, path_error, path_loss
from . import pipeline as PL


def run_dir(cfg, variant, seed):
    folder = cfg.P.out / "models" / f"{variant}_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---------------------------------------------------------------- data on the device
def load_volumes(cfg, subject_ids, protocols, device):
    """Per subject: inputs for every training protocol, reference fODF and anatomy on the device; V1-V3 surface
    seeds stay on the CPU, where patches are drawn."""
    volumes, patch = [], cfg.patch
    for sid in subject_ids:
        session = PL.session(sid)
        anat = PL.load_anat(cfg, session)
        padding = [max(0, patch - int(length)) for length in anat["shape"]]  # make every axis at least one patch long
        pad_volume = lambda array: F.pad(
            torch.as_tensor(np.asarray(array, np.float32)), (0, padding[2], 0, padding[1], 0, padding[0])
        )
        seed_vertices = np.flatnonzero(anat["area"] > 0)
        volumes.append(
            dict(
                sid=str(sid),
                x={
                    name: pad_volume(PL.net_input(cfg, session, name, anat, cfg.model.input)).half().to(device)
                    for name in protocols
                },
                y=pad_volume(PL.fodf(cfg, session, "REF", "msmt_csd", anat=anat)["sh"]).to(device),
                anat=pad_volume(PL.anat_channels(anat)).to(device),
                seeds=torch.as_tensor(
                    anat["verts_vox"][seed_vertices]
                    - cfg.tracking.seed_depth_mm / cfg.ref_voxel * anat["normals_vox"][seed_vertices]
                ).float(),
                inward=torch.as_tensor(-anat["normals_vox"][seed_vertices]).float(),
            )
        )
        volumes[-1]["wm_idx"] = (volumes[-1]["anat"][0] > 0.9).nonzero().float().cpu()
    return volumes


def sample_batch(cfg, volumes, protocols, batch_size, generator, device):
    """B random cubic patches (random subject and protocol). 70% are centred near a V1-V3 seed (within +-patch/4),
    the rest anywhere. Returns stacked x, y, anat, path seeds and inward normals (B, S, 3) in patch-voxel units
    (at least 2 voxels from the border), and valid (B, S): all False when the patch holds no usable seed."""
    patch, n_seeds, settings = cfg.patch, cfg.model.path.seeds_per_patch, cfg.model
    inputs, targets, anatomy, seed_points, inward_normals, valid = [], [], [], [], [], []
    for _ in range(batch_size):  # positions are drawn on the CPU; only the patches are cut on the device
        volume = volumes[int(torch.randint(len(volumes), (1,), generator=generator))]
        name = protocols[int(torch.randint(len(protocols), (1,), generator=generator))]
        size = torch.tensor(volume["y"].shape[1:])
        if torch.rand(1, generator=generator) < 0.7 and len(volume["seeds"]):  # most patches sit on visual cortex
            centre = (
                volume["seeds"][int(torch.randint(len(volume["seeds"]), (1,), generator=generator))]
                + (torch.rand(3, generator=generator) - 0.5) * patch / 2
            )
        else:
            centre = torch.rand(3, generator=generator) * size
        origin = torch.minimum(torch.clamp((centre - patch / 2).round().long(), min=0), size - patch)
        window = (slice(None),) + tuple(slice(start, start + patch) for start in origin.tolist())
        inputs.append(volume["x"][name][window].float())
        targets.append(volume["y"][window])
        anatomy.append(volume["anat"][window])
        points = (volume["wm_idx"] if settings.seed_where == "wm" else volume["seeds"]) - origin
        usable = ((points > 2) & (points < patch - 3)).all(1).nonzero()[:, 0]
        if len(usable):
            picked = usable[torch.randint(len(usable), (n_seeds,), generator=generator)]
            inward = (
                F.normalize(torch.randn(n_seeds, 3, generator=generator), dim=-1)
                if settings.seed_where == "wm"
                else volume["inward"][picked]
            )  # wm: ablation
            seed_points.append(points[picked])
            inward_normals.append(inward)
            valid.append(torch.ones(n_seeds, dtype=torch.bool))
        else:
            seed_points.append(torch.full((n_seeds, 3), patch / 2.0))
            inward_normals.append(torch.zeros(n_seeds, 3))
            valid.append(torch.zeros(n_seeds, dtype=torch.bool))
    return [torch.stack(parts) for parts in (inputs, targets, anatomy)] + [
        torch.stack(parts).to(device) for parts in (seed_points, inward_normals, valid)
    ]


def net_in(cfg, inputs, anat):
    return torch.cat([inputs, anat], 1) if cfg.model.use_anat else inputs


# ---------------------------------------------------------------- loss
def losses(cfg, model, tracer, batch, steps, scale, amp_dev):
    """Returns (total loss, {part: float}). Voxel MSE (inside wm > wm_mask_thr) and the non-negativity penalty are
    divided by scale**2 and scale (std of the training fODF coefficients); the path loss (mm) is added as a
    fraction of the path length steps * step_mm."""
    inputs, target, anat, seeds, inward, valid = batch
    settings = cfg.model
    with torch.autocast(amp_dev, enabled=bool(settings.amp) and amp_dev == "cuda"):
        pred = model(net_in(cfg, inputs, anat))
    pred = pred.float()
    wm = anat[:, :1]
    mask = (wm > cfg.wm_mask_thr).float()
    parts = {
        "voxel": ((pred - target) ** 2 * mask).sum() / (mask.sum() * target.shape[1] + 1e-8) / scale**2,
        "nonneg": nonneg_loss(tracer, pred, wm) / scale,
    }
    total = parts["voxel"] + settings.lambda_nonneg * parts["nonneg"]
    if settings.lambda_path > 0:
        step_vox = settings.path.step_mm / cfg.ref_voxel
        parts["path_mm"] = path_loss(
            tracer, pred, target, wm, seeds, inward, valid, steps, step_vox, cfg.ref_voxel, settings.path.one_step
        )
        total = total + settings.lambda_path * parts["path_mm"] / (
            steps * settings.path.step_mm
        )  # error as a fraction of path length
    return total, {name: float(value.detach()) for name, value in parts.items()}


@torch.no_grad()
def predict(cfg, model, inputs, anat, device):
    """Whole-crop inference. x: (C, X, Y, Z) numpy network input; anat: 7-channel numpy. Returns (45, X, Y, Z) float32."""
    model.eval()
    tensor = torch.as_tensor(
        np.concatenate([inputs.astype(np.float32), anat]) if cfg.model.use_anat else inputs.astype(np.float32)
    )[None].to(device)
    multiple = 2 ** (cfg.model.levels - 1)
    padding = [(-length) % multiple for length in tensor.shape[2:]]
    tensor = F.pad(tensor, (0, padding[2], 0, padding[1], 0, padding[0]))
    with torch.autocast(device.type, enabled=bool(cfg.model.amp) and device.type == "cuda"):
        output = model(tensor)
    crop_shape = inputs.shape[1:]
    return output[0, :, : crop_shape[0], : crop_shape[1], : crop_shape[2]].float().cpu().numpy()


def auto_batch(cfg, model, tracer, volumes, protocols, device, scale, resumed=None, shared=None):
    """Largest batch (8, 16, 32, 64) whose step fits in 70% of GPU memory (4 if none; 2 off CUDA). Searched once
    with the `path` network and saved to `shared`, so every run uses the same batch; `resumed` keeps a run's own."""
    if cfg.batch_size != "auto":
        return int(cfg.batch_size)
    if resumed:
        return int(resumed)
    if device.type != "cuda":
        return 2
    if shared and shared.exists():
        return int(load_json(shared)["batch"])
    benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    best, generator = 4, torch.Generator().manual_seed(0)
    lambda_path = cfg.model.lambda_path
    cfg.model["lambda_path"] = 1.0
    held = peak = torch.cuda.memory_allocated(device)  # volumes and weights
    limit = 0.7 * torch.cuda.mem_get_info(device)[1]
    for size in (8, 16, 32, 64):
        if held + 2 * (peak - held) > limit:
            break
        try:
            torch.cuda.reset_peak_memory_stats(device)
            total, _ = losses(
                cfg,
                model,
                tracer,
                sample_batch(cfg, volumes, protocols, size, generator, device),
                cfg.model.path.steps_end,
                scale,
                "cuda",
            )
            total.backward()
            model.zero_grad(set_to_none=True)
            del total
            peak = torch.cuda.max_memory_allocated(device)
            if peak > limit:
                break
            best = size
        except torch.OutOfMemoryError:
            break
    cfg.model["lambda_path"] = lambda_path
    torch.backends.cudnn.benchmark = benchmark
    torch.cuda.empty_cache()
    if shared:
        try:  # the first run to finish the search sets it
            descriptor = os.open(shared, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            with os.fdopen(descriptor, "w") as handle:
                handle.write(f'{{"batch": {best}}}')
        except FileExistsError:
            pass
        best = int(load_json(shared)["batch"])
    return best


# ---------------------------------------------------------------- training loop
def train(cfg, variant="path", seed=0, subjects=None):
    """Train one (variant, seed) into outputs/models/<variant>_s<seed>/ (last.pt, best.pt, log.json); resumes from last.pt.
    Progress = max(step / (epochs * steps_per_epoch), hours / max_train_hours); it drives the cosine learning rate
    and the path-length curriculum, and training ends at 1. SIGINT/SIGTERM save a checkpoint and exit with 130."""
    reference_cfg = copy.deepcopy(cfg)
    reference_kind = apply_variant(reference_cfg, "path")  # the network the batch size is set with
    kind = apply_variant(cfg, variant)
    distributed = "LOCAL_RANK" in os.environ
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if distributed else (0, 1)
    if distributed:
        device = (
            torch.device("cuda", int(os.environ["LOCAL_RANK"])) if torch.cuda.is_available() else torch.device("cpu")
        )
        if device.type == "cuda":
            torch.cuda.set_device(device)
        torch.distributed.init_process_group("nccl" if device.type == "cuda" else "gloo")
    else:
        device = get_device()
    if not cfg.dev_mode and device.type != "cuda":  # days on the CPU instead of hours on the GPU
        raise SystemExit(
            "the full configuration trains on a CUDA GPU and torch finds none: check nvidia-smi and that "
            "the CUDA build of torch is installed (python -c 'import torch; print(torch.cuda.is_available())')"
        )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    seed_all(cfg.seed + 1000 * seed + rank)
    folder = run_dir(cfg, variant, seed)
    checkpoint = folder / "last.pt"
    take_over(folder / "pid", owner=rank == 0)
    previous = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint.exists() else None
    total_steps = cfg.epochs * cfg.steps_per_epoch
    max_hours = cfg.max_train_hours
    if previous and previous["meta"].get("steps", total_steps) != total_steps:  # every network trains equally long
        raise SystemExit(
            f"{folder.name} was started with {previous['meta']['steps']} training steps and config.yaml now gives "
            f"{total_steps}; delete {folder} to train it again"
        )
    if previous and previous["state"]["done"]:
        return folder  # this run already finished
    if previous is None and rank == 0:
        clear_outputs(cfg, variant, seed)
    subjects = subjects or load_json(cfg.P.out / "subjects.json")
    protocols = list(cfg.train_protocols)
    train_volumes = load_volumes(cfg, subjects["train"], protocols, device)
    val_volumes = load_volumes(cfg, subjects["val"], protocols, device)
    scale = float(
        torch.cat([volume["y"][:, volume["anat"][0] > 0.5].flatten() for volume in train_volumes]).std()
    )  # fODF coefficient scale

    model = build_model(cfg, kind).to(device)
    if cfg.model.compile:
        model = torch.compile(model)
    tracer = PathTracer(cfg.sh_order_out, cfg.model.path.cone_deg, cfg.model.path.sharpen).to(device)
    shared_batch = cfg.P.out / "models" / "batch.json"
    if previous or cfg.batch_size != "auto" or device.type != "cuda" or shared_batch.exists():
        batch_size = auto_batch(
            cfg,
            model,
            tracer,
            train_volumes,
            protocols,
            device,
            scale,
            previous["meta"]["batch"] if previous else None,
            shared_batch,
        )
    else:
        probe = build_model(reference_cfg, reference_kind).to(device)  # the batch is set with the path network
        probe_volumes = (
            train_volumes
            if reference_cfg.model.input == cfg.model.input
            else load_volumes(reference_cfg, subjects["train"], protocols, device)
        )
        batch_size = auto_batch(
            reference_cfg, probe, tracer, probe_volumes, protocols, device, scale, None, shared_batch
        )
        del probe, probe_volumes
        torch.cuda.empty_cache()
    seed_all(cfg.seed + 1000 * seed + rank)  # the batch search draws random numbers; start every run alike
    network = (
        torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
        if distributed
        else model
    )
    base_lr = cfg.model.lr * world
    optimizer = torch.optim.AdamW(network.parameters(), lr=base_lr, weight_decay=cfg.model.weight_decay)
    scaler = torch.amp.GradScaler(enabled=bool(cfg.model.amp) and device.type == "cuda")
    state = dict(step=0, hours=0.0, log=[], done=False)
    if previous:  # resume
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["opt"])
        scaler.load_state_dict(previous["scaler"])
        state = previous["state"]
    # a resumed run restores its random streams; under torchrun only rank 0 saved them, so the others restart
    generator = torch.Generator().manual_seed(cfg.seed + 1000 * seed + rank + (state["step"] if distributed else 0))
    if previous and previous.get("rng") and not distributed:
        saved_rng = previous["rng"]
        generator.set_state(saved_rng["gen"])
        torch.set_rng_state(saved_rng["torch"])
        np.random.set_state(saved_rng["numpy"])
        random.setstate(saved_rng["python"])
        if device.type == "cuda" and saved_rng.get("cuda") is not None:
            torch.cuda.set_rng_state(saved_rng["cuda"], device)
    val_generator = lambda: torch.Generator().manual_seed(123)
    start_time = last_checkpoint = time.time()
    hours_before = state["hours"]
    path_settings = cfg.model.path

    def save(done=False):
        state["hours"], state["done"] = hours_before + (time.time() - start_time) / 3600, done
        rng_state = dict(
            gen=generator.get_state(),
            torch=torch.get_rng_state(),
            numpy=np.random.get_state(),
            python=random.getstate(),
            cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        )
        atomic(
            checkpoint,
            lambda handle: torch.save(
                dict(
                    model=model.state_dict(),
                    opt=optimizer.state_dict(),
                    scaler=scaler.state_dict(),
                    state=state,
                    rng=rng_state,
                    meta=meta(),
                ),
                handle,
            ),
        )
        save_json(state["log"], folder / "log.json")

    def meta():
        return dict(
            variant=variant, seed=seed, kind=kind, batch=batch_size, world=world, scale=scale, steps=total_steps
        )

    def save_best(row):
        """best.pt: the weights with the lowest validation path error so far (validation subjects only), the model
        that inference uses. Every variant is selected on the same number."""
        if row["val_path_mm"] >= state.get("best_val_path_mm", math.inf):
            return
        state["best_val_path_mm"], state["best_step"] = row["val_path_mm"], row["step"]
        atomic(
            folder / "best.pt",
            lambda handle: torch.save(
                dict(model=model.state_dict(), meta=dict(meta(), step=row["step"], val_path_mm=row["val_path_mm"])),
                handle,
            ),
        )

    stop = {"now": False}  # Ctrl-C / kernel interrupt / kill: save first, then leave
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, lambda *_: stop.update(now=True))
    if rank == 0 and state["step"]:
        print(
            f"resuming {variant}_s{seed} at step {state['step']} ({state['hours']:.2f} h done, batch {batch_size})",
            flush=True,
        )

    parent = os.getppid()
    while state["step"] < total_steps:
        hours = hours_before + (time.time() - start_time) / 3600
        if os.getppid() != parent:  # the notebook or shell that started this run is gone
            stop["now"] = True
        if distributed:  # rank 0 decides, so all ranks stop at the same step
            control = torch.tensor([float(stop["now"]), hours], device=device)
            torch.distributed.broadcast(control, 0)
            stop["now"], hours = bool(control[0]), float(control[1])
        if stop["now"]:
            if rank == 0:
                save()
                print(f"stopped at step {state['step']}; checkpoint saved, run again to continue", flush=True)
            raise SystemExit(130)
        progress = max(state["step"] / total_steps, hours / max_hours if max_hours else 0.0)
        if progress >= 1:
            break
        path_steps = int(
            round(
                path_settings.steps_start
                + (path_settings.steps_end - path_settings.steps_start) * min(1.0, 2 * progress)
            )
        )  # path-length curriculum
        for group in optimizer.param_groups:
            group["lr"] = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        network.train()
        loss, parts = losses(
            cfg,
            network,
            tracer,
            sample_batch(cfg, train_volumes, protocols, batch_size, generator, device),
            path_steps,
            scale,
            device.type,
        )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        state["step"] += 1
        if rank == 0 and state["step"] % cfg.steps_per_epoch == 0:
            network.eval()
            val_parts = []
            with torch.no_grad():
                # every variant reports the same validation numbers: voxel loss and accumulated path error
                val_gen = val_generator()
                saved = cfg.model.lambda_path, path_settings.one_step
                cfg.model["lambda_path"], path_settings["one_step"] = 1.0, False
                for _ in range(8):
                    val_parts.append(
                        losses(
                            cfg,
                            model,
                            tracer,
                            sample_batch(cfg, val_volumes, protocols, batch_size, val_gen, device),
                            path_settings.steps_end,
                            scale,
                            device.type,
                        )[1]
                    )
                cfg.model["lambda_path"], path_settings["one_step"] = saved
            row = dict(
                step=state["step"],
                hours=round(hours, 4),
                hours_left=round(
                    min(
                        hours / state["step"] * (total_steps - state["step"]),
                        max_hours - hours if max_hours else math.inf,
                    ),
                    2,
                ),
                path_steps=path_steps,
                **{f"train_{key}": value for key, value in parts.items()},
                **{f"val_{key}": float(np.mean([part[key] for part in val_parts])) for key in val_parts[0]},
            )
            if device.type == "cuda":  # peak since the last row; reserved far above used = fragmentation
                row["gpu_gb"], row["gpu_reserved_gb"] = (
                    round(value / 2**30, 1)
                    for value in (torch.cuda.max_memory_allocated(device), torch.cuda.memory_reserved(device))
                )
                torch.cuda.reset_peak_memory_stats(device)
            state["log"].append(row)
            save_best(row)
            print(
                f"step {state['step']}/{total_steps} ({100 * state['step'] / total_steps:.0f}%) | {hours:.2f} h done, "
                f"{row['hours_left']} h left | "
                + " ".join(
                    f"{key} {value:.4g}" for key, value in row.items() if key.startswith(("train_", "val_", "gpu_"))
                ),
                flush=True,
            )
        if rank == 0 and time.time() - last_checkpoint > cfg.model.ckpt_minutes * 60:
            save()
            last_checkpoint = time.time()
    if rank == 0:
        save(done=True)
    if distributed:
        torch.distributed.destroy_process_group()
    return folder


def take_over(lock, owner=True, wait=600):
    """One process per run folder: a live earlier process named in `lock` ("host pid") is asked to save and stop,
    and this call waits for it (torchrun ranks excepted). owner: write the lock."""
    host, own_pid = socket.gethostname(), os.getpid()
    if lock.exists():
        host_name, _, pid = lock.read_text().partition(" ")
        pid = int(pid) if pid.strip().isdigit() else -1
        alive = lambda: _alive(pid)
        try:
            rival = host_name == host and pid not in (-1, own_pid) and alive() and os.getpgid(pid) != os.getpgid(0)
        except ProcessLookupError:  # it exited between the two checks
            rival = False
        if rival:
            if owner:
                print(f"another process ({pid}) is training this run; asking it to save and stop", flush=True)
                os.kill(pid, signal.SIGINT)
            start_time = time.time()
            while alive() and time.time() - start_time < wait:
                time.sleep(1)
            if alive():
                raise RuntimeError(f"process {pid} still holds {lock.parent}; stop it and run again")
    if owner:
        lock.write_text(f"{host} {own_pid}")
        atexit.register(lambda: lock.exists() and lock.read_text() == f"{host} {own_pid}" and lock.unlink())


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def clear_outputs(cfg, variant, seed):
    """Remove cached fODFs, tractograms and scores of an earlier network with this name, so that a retrained
    network is never mixed with results of the old one."""
    pattern = re.compile(rf"_net_{re.escape(variant)}_s{seed}([._+]|$)")
    for path in (cfg.P.out / "cache").glob(f"*/*_net_{variant}_s{seed}*"):
        if pattern.search(path.name):
            path.unlink()


def load_trained(cfg, variant, seed, device):
    """Returns (model, cfg adjusted to the variant)."""
    kind = apply_variant(cfg, variant)
    best = run_dir(cfg, variant, seed) / "best.pt"
    if not best.exists():
        raise FileNotFoundError(
            f"{best}: no best-on-validation checkpoint (runs trained before it was added); delete "
            f"{best.parent} and train it again"
        )
    checkpoint = torch.load(best, map_location=device, weights_only=False)
    model = build_model(cfg, kind).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["meta"]


def net_fodf(cfg, session, name, method, model, device):
    """Run a trained network on protocol `name` and cache its fODF under `method` (e.g. net_path_s0)."""
    path = PL.sdir(cfg, session, method) / f"fodf_{name}_{method}.npz"
    if path.exists():
        return
    anat = PL.load_anat(cfg, session)
    sh = predict(cfg, model, PL.net_input(cfg, session, name, anat, cfg.model.input), PL.anat_channels(anat), device)
    sh = sh * (anat["pve"][0] > cfg.wm_mask_thr)
    PL.save_fodf(cfg, session, name, method, sh, anat["affine"], anat["pve"])


def net_fodf_all(cfg, sessions, names, method, model, device, desc=None):
    """net_fodf for every session and protocol in `names`. The network inputs are made first, one session per core;
    then cfg.gpu_jobs sessions at a time run in threads that share the model, because reading the inputs and
    writing the fODFs take longer than the forward pass."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import tqdm

    todo = [
        session
        for session in sessions
        if not all((PL.sdir(cfg, session, method) / f"fodf_{name}_{method}.npz").exists() for name in names)
    ]
    PL.inputs_for(cfg, todo, names, cfg.model.input, f"{method}: inputs")

    def predict_session(session):
        for name in names:
            net_fodf(cfg, session, name, method, model, device)

    with ThreadPoolExecutor(max(1, int(cfg.gpu_jobs))) as executor:
        list(tqdm(executor.map(predict_session, sessions), total=len(sessions), desc=desc or method))


def path_errors(cfg, sessions, combos, name, device):
    """Path error (model.path_error) of every session x (protocol, method) against the reference fODF, with the
    tracer and path length of the training loss and one seed per V1-V3 vertex. Cached per file; writes
    outputs/results/<name>.csv and returns the table."""
    import pandas as pd
    from tqdm.auto import tqdm

    settings = cfg.model.path
    signature = json.dumps(
        [settings.cone_deg, settings.sharpen, settings.steps_end, settings.step_mm, cfg.tracking.seed_depth_mm]
    )
    tracer = PathTracer(cfg.sh_order_out, settings.cone_deg, settings.sharpen).to(device)
    on_device = lambda array: torch.as_tensor(np.asarray(array, np.float32))[None].to(device)
    rows = []
    for session in tqdm(sessions, desc=name):
        files = {combo: PL.sdir(cfg, session, combo[1]) / f"patherr_{combo[0]}_{combo[1]}.json" for combo in combos}
        cached = {combo: load_json(file) for combo, file in files.items() if file.exists()}
        cached = {combo: row for combo, row in cached.items() if row.get("settings") == signature}
        if len(cached) < len(combos):
            anat = PL.load_anat(cfg, session)
            vertices = np.flatnonzero(anat["area"] > 0)
            seeds = on_device(
                anat["verts_vox"][vertices] - cfg.tracking.seed_depth_mm / cfg.ref_voxel * anat["normals_vox"][vertices]
            )
            inward = on_device(-anat["normals_vox"][vertices])
            reference = on_device(PL.fodf(cfg, session, "REF", "msmt_csd", anat=anat)["sh"])
            wm = on_device(anat["pve"][0])[None]
        for protocol, method in combos:
            if (protocol, method) not in cached:
                predicted = on_device(PL.on_ref_grid(PL.fodf(cfg, session, protocol, method, anat=anat), anat))
                mean, end, n_paths = path_error(
                    tracer,
                    predicted,
                    reference,
                    wm,
                    seeds,
                    inward,
                    settings.steps_end,
                    settings.step_mm / cfg.ref_voxel,
                    cfg.ref_voxel,
                )
                cached[(protocol, method)] = dict(
                    sid=session["sid"],
                    protocol=protocol,
                    method=method,
                    path_err_mm=mean,
                    path_end_err_mm=end,
                    n_paths=n_paths,
                    settings=signature,
                )
                save_json(cached[(protocol, method)], files[(protocol, method)])
            rows.append({key: value for key, value in cached[(protocol, method)].items() if key != "settings"})
    table = pd.DataFrame(rows)
    folder = cfg.P.out / "results"
    folder.mkdir(exist_ok=True)
    table.to_csv(folder / f"{name}.csv", index=False)
    return table


def trained_runs(cfg):
    """[(variant, seed)] of the runs listed in config.yaml that have finished training."""
    from .jobs import train_commands

    runs = []
    for name, _ in train_commands(cfg):
        checkpoint = cfg.P.out / "models" / name / "last.pt"
        if checkpoint.exists() and torch.load(checkpoint, map_location="cpu", weights_only=False)["state"]["done"]:
            variant, seed = name.rsplit("_s", 1)
            runs.append((variant, int(seed)))
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="path", choices=list(VARIANTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev", action="store_true")
    arguments = parser.parse_args()
    train(load_config(arguments.dev), arguments.variant, arguments.seed)
