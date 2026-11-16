"""Cached steps per session (one scan: 3T, 7T, retest, nsd, nsd2): crop -> protocol -> fODF -> tractogram -> scores.
A method is the fODF name plus an optional tracker suffix (csd, net_path_s0, csd+prob); no suffix = particle filter.
"""

import copy, hashlib, itertools, json, shutil, threading
import numpy as np
from scipy.spatial import cKDTree
from . import metrics as M, nsd
from .atlas import atlas_files, atlas_key_source
from .core import atomic, load_json, n_jobs, save_json
from .degrade import make_protocol, to_grid
from .download import (
    HCP_BUCKET,
    anon_client,
    hcp_client,
    hcp_download,
    hcp_figure_keys,
    hcp_keys,
    osf_download,
    s3_dirs,
    s3_exists,
    s3_size,
    s3_get_many,
    update_manifest,
    remove_stale_parts,
)
from .recon import fit_fodf, signal_sh
from .subject import build_anat, load_dwi_crop, load_prf, prf_subject_ids, subject_dir
from .track import area_pairs, density_map, endpoints, make_seeds, run_tracking

CLASSICAL = {"csd": "msmt_csd", "rumba": "rumba"}  # label -> recon method ("csd" is multi-shell when it can be)


def split_method(method):
    """'csd+prob' -> ('csd', 'prob'); 'csd' -> ('csd', 'pft')."""
    base, _, tracker = method.partition("+")
    return base, tracker or "pft"


def session(sid, kind="3T"):
    """dict(sid, kind, release, dwi folder name, tag = cache folder name: sid for 3T, else sid_kind)."""
    return dict(
        sid=str(sid),
        kind=kind,
        release="HCP_Retest" if kind == "retest" else "HCP_1200",
        dwi="Diffusion_7T" if kind == "7T" else "Diffusion",
        tag=str(sid) if kind == "3T" else f"{sid}_{kind}",
    )


def cfg_for(cfg, session):
    """Settings for this session: NSD has its own seed density and eccentricity limit (config.yaml, `nsd:`)."""
    if not session["kind"].startswith("nsd"):
        return cfg
    adjusted = copy.deepcopy(cfg)
    adjusted["seeds_per_vertex"] = cfg.nsd.seeds_per_vertex
    adjusted.prf["ecc_max"] = cfg.nsd.ecc_max
    return adjusted


def sdir(cfg, session, method=""):
    """Cache folder of a session; files of trained networks (net_*) go under the run's own output folder."""
    root = cfg.P.out / "cache" if method.startswith("net_") else cfg.P.cache
    folder = root / session["tag"]
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---------------------------------------------------------------- one session per core
def each(cfg, function, items, desc=None, gb=2.0):
    """[function(cfg, item) for item in items], one item per worker process (at most one per `gb` GB of free
    memory); spare cores go to each worker's own fits."""
    import psutil
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    items = list(items)
    total = n_jobs(cfg)
    if not items:
        return []
    workers = max(1, min(total, len(items), int(psutil.virtual_memory().available / 2**30 / gb)))
    worker_cfg = copy.deepcopy(cfg)
    worker_cfg["n_jobs"] = max(1, total // workers)
    run_one = lambda index, item: (index, function(worker_cfg, item))
    out = [None] * len(items)
    for index, result in tqdm(
        Parallel(n_jobs=workers, return_as="generator_unordered")(
            delayed(run_one)(index, item) for index, item in enumerate(items)
        ),
        total=len(items),
        desc=desc,
    ):
        out[index] = result
    return out


# ---------------------------------------------------------------- crop
def prepare(cfg, session, s3=None):
    """Download (if needed) and cache the occipital crop of one session."""
    folder = sdir(cfg, session)
    if (folder / "anat.npz").exists() and (folder / "dwi.npz").exists():
        return folder
    if session["kind"].startswith("nsd"):  # nsd = first diffusion run, nsd2 = second run
        run = 2 if session["kind"] == "nsd2" else 1
        nsd.nsd_download(cfg, session["sid"], run)
        anat = nsd.build_anat(cfg, session["sid"], run)
        dwi, bvals, bvecs, mask = nsd.load_dwi_crop(cfg, session["sid"], anat, run)
        atomic(folder / "dwi.npz", lambda temporary: np.savez(temporary, dwi=dwi, bvals=bvals, bvecs=bvecs, mask=mask))
        atomic(folder / "anat.npz", lambda temporary: np.savez(temporary, **anat))
        return folder
    raw = subject_dir(cfg, session["sid"], session["release"])
    if not (raw / session["dwi"] / "data.nii.gz").exists() or not (raw / "aparc+aseg.nii.gz").exists():
        if session["kind"] != "7T" or not (raw / "aparc+aseg.nii.gz").exists():
            hcp_download(cfg, session["sid"], "3T", session["release"], s3)  # anatomy + 3T diffusion
        if session["kind"] == "7T":
            hcp_download(cfg, session["sid"], "7T", session["release"], s3)
    anat = build_anat(cfg, session["sid"], session["release"], session["dwi"])
    dwi, bvals, bvecs, mask = load_dwi_crop(cfg, session["sid"], anat, session["release"], session["dwi"])
    atomic(folder / "dwi.npz", lambda temporary: np.savez(temporary, dwi=dwi, bvals=bvals, bvecs=bvecs, mask=mask))
    atomic(folder / "anat.npz", lambda temporary: np.savez(temporary, **anat))
    if not cfg.keep_raw:
        shutil.rmtree(raw / session["dwi"], ignore_errors=True)
    return folder


def prepare_all(cfg, sessions, desc="crops"):
    """prepare() for every session, in parallel: unpacking a scan takes one core for several seconds. 7T sessions
    come after the others, because a 7T crop uses the anatomy files its 3T session may still be fetching."""
    each(cfg, prepare, [session for session in sessions if session["kind"] != "7T"], desc)
    each(cfg, prepare, [session for session in sessions if session["kind"] == "7T"], f"{desc} (7T)")


def load_anat(cfg, session):
    anat = dict(np.load(sdir(cfg, session) / "anat.npz"))
    anat["sid"], anat["hemis"] = str(anat["sid"]), str(anat["hemis"])
    return anat


def load_dwi(cfg, session):
    stored = np.load(sdir(cfg, session) / "dwi.npz")
    return stored["dwi"], stored["bvals"], stored["bvecs"], stored["mask"]


def group_key_source(cfg, anat, sid="999999"):
    """Group-average retinotopy on this subject's vertices (template key)."""
    if "tpl_ang" in anat:  # NSD: carried over through the fsaverage sphere
        return {f"prf_{quantity}": anat[f"tpl_{quantity}"] for quantity in ("ang", "ecc", "r2")}
    prf = load_prf(cfg, sid)
    return {
        f"prf_{quantity}": np.concatenate([prf[hemi][quantity] for hemi in anat["hemis"]])
        for quantity in ("ang", "ecc", "r2")
    }


# ---------------------------------------------------------------- protocol
def protocol(cfg, session, name, anat=None):
    """Simulated acquisition `name` (config.yaml) or the scan as acquired (name == 'NATIVE')."""
    anat = anat if anat is not None else load_anat(cfg, session)
    dwi, bvals, bvecs, _ = load_dwi(cfg, session)
    if name == "NATIVE":
        return dict(dwi=dwi, bvals=bvals, bvecs=bvecs, affine=anat["affine"], pve=anat["pve"])
    return make_protocol(
        dwi,
        bvals,
        bvecs,
        anat["affine"],
        anat["pve"],
        cfg.protocols[name],
        cfg.n_b0,
        seed=cfg.seed + int(session["sid"]) % 100000,
    )


def anat_channels(anat):
    """7 anatomical input channels: wm, gm, csf, signed distance / 5 mm, inward normal (3)."""
    return np.concatenate([anat["pve"], np.clip(anat["sdist"] / 5.0, -2, 2)[None], anat["normal_field"]]).astype(
        np.float32
    )


def net_input(cfg, session, name, anat=None, source="signal"):
    """Cached float16 network input of protocol `name` on the crop grid: the CSD fODF (source = fodf) or the
    per-shell signal SH (source = signal)."""
    path = sdir(cfg, session) / (f"in_{name}.npy" if source == "signal" else f"in_{name}_fodf.npy")
    if path.exists():
        return np.load(path)
    anat = anat if anat is not None else load_anat(cfg, session)
    if source == "fodf":
        values = on_ref_grid(fodf(cfg, session, name, "csd", anat=anat), anat).astype(np.float16)
    else:
        simulated = protocol(cfg, session, name, anat)
        values = signal_sh(simulated["dwi"], simulated["bvals"], simulated["bvecs"], cfg.shells_in, cfg.sh_order_in)
        values = np.moveaxis(to_grid(np.moveaxis(values, 0, -1), anat["shape"]), -1, 0).astype(np.float16)
    atomic(path, lambda temporary: np.save(temporary, values))
    return values


def _net_inputs(cfg, sid):
    from .model import input_sources

    for name in cfg.train_protocols:
        for source in sorted(input_sources(cfg)):
            net_input(cfg, session(sid), name, source=source)


def _inputs_of(cfg, job):
    session, names, source = job
    for name in names:
        net_input(cfg, session, name, source=source)


def inputs_for(cfg, sessions, names, source="signal", desc="network inputs"):
    """Cached network inputs of these sessions and protocols, one session per worker."""
    input_file = lambda session, name: sdir(cfg, session) / (
        f"in_{name}.npy" if source == "signal" else f"in_{name}_fodf.npy"
    )
    each(
        cfg,
        _inputs_of,
        [
            (session, names, source)
            for session in sessions
            if not all(input_file(session, name).exists() for name in names)
        ],
        desc,
    )


def net_inputs_all(cfg, sids):
    """Cached network inputs of every training protocol for these subjects, one subject per worker."""
    each(cfg, _net_inputs, sids, "network inputs")


# ---------------------------------------------------------------- fODF
def fodf(cfg, session, name, method, device="cpu", anat=None):
    """fODF of `method` on protocol `name`. Returns dict(sh, affine, pve). Methods: msmt_csd (reference), csd, rumba,
    csd_mppca (denoise first), csd_interp (interpolate the DWIs to the reference grid first)."""
    path = sdir(cfg, session, method) / f"fodf_{name}_{method}.npz"
    if path.exists():
        stored = np.load(path)
        return dict(sh=stored["sh"].astype(np.float32), affine=stored["affine"], pve=stored["pve"])
    if method.startswith("net_"):
        raise FileNotFoundError(f"{path.name}: network outputs are written by notebook 06 (train.net_fodf)")
    anat = anat if anat is not None else load_anat(cfg, session)
    simulated = protocol(cfg, session, name, anat)
    if method == "csd_mppca":
        from dipy.denoise.localpca import mppca

        simulated["dwi"] = mppca(simulated["dwi"], patch_radius=2).astype(np.float32)
    if method == "csd_interp":
        simulated.update(dwi=to_grid(simulated["dwi"], anat["shape"]), affine=anat["affine"], pve=anat["pve"])
    recon = cfg.ref_model if method == "msmt_csd" else CLASSICAL.get(method.split("_")[0], "msmt_csd")
    sh = fit_fodf(simulated, simulated["pve"][0] > cfg.wm_mask_thr, recon, cfg, n_jobs(cfg), device)
    return save_fodf(cfg, session, name, method, sh, simulated["affine"], simulated["pve"])


def _fodf_file(cfg, job):
    session, name, method, device = job
    fodf(cfg, session, name, method, device)


def _gpu_fit(cfg, name, method, device):
    """True for the fits that have a GPU solver: multi-shell CSD and RUMBA-SD, on CUDA."""
    recon = cfg.ref_model if method == "msmt_csd" else CLASSICAL.get(method.split("_")[0], "msmt_csd")
    multi = name == "NATIVE" or len(cfg.protocols[name]["shells"]) > 1
    return str(device).startswith("cuda") and (recon == "rumba" or (recon == "msmt_csd" and multi))


def fodf_all(cfg, sessions, name, method, device="cpu", desc="fODF"):
    """fodf() for every session: GPU fits one after another in this process, the others one session per core."""
    todo = [session for session in sessions if not (sdir(cfg, session, method) / f"fodf_{name}_{method}.npz").exists()]
    if _gpu_fit(cfg, name, method, device):
        from tqdm.auto import tqdm

        for session in tqdm(todo, desc=desc):
            fodf(cfg, session, name, method, device)
    else:
        each(cfg, _fodf_file, [(session, name, method, "cpu") for session in todo], desc)


def save_fodf(cfg, session, name, method, sh, affine, pve):
    """Store the fODF as float16 and return it as read back, so new and cached fits track alike."""
    sh, pve = sh.astype(np.float16), pve.astype(np.float16)
    atomic(
        sdir(cfg, session, method) / f"fodf_{name}_{method}.npz",
        lambda temporary: np.savez(temporary, sh=sh, affine=affine, pve=pve),
    )
    return dict(sh=sh.astype(np.float32), affine=affine, pve=pve)


def on_ref_grid(fodf_volume, anat):
    """SH volume resampled to the crop grid (for voxel-wise comparison with the reference)."""
    if tuple(fodf_volume["sh"].shape[1:]) == tuple(anat["shape"]):
        return fodf_volume["sh"]
    return np.moveaxis(to_grid(np.moveaxis(fodf_volume["sh"], 0, -1), anat["shape"]), -1, 0)


# ---------------------------------------------------------------- tractogram
def tract(cfg, session, name, method, device="cpu", anat=None, tag=""):
    """V1-V3 tractogram: end vertices, counts, density map and streamlines; same seeds for every method. Tracked
    again when the stored tracking settings differ. tag: cache-file suffix for other tracking settings."""
    path = sdir(cfg, session, method) / f"tract_{name}_{method}{tag}.npz"
    cfg = cfg_for(cfg, session)
    settings = json.dumps([dict(cfg.tracking), cfg.seeds_per_vertex], sort_keys=True)
    if path.exists():
        cached = dict(np.load(path))
        if str(cached.get("settings")) == settings:
            return cached
    anat = anat if anat is not None else load_anat(cfg, session)
    seeds, _ = make_seeds(anat, cfg, cfg.seed)
    if method == "proximity":
        out = proximity_tract(anat, cfg)
    else:
        base, tracker = split_method(method)
        # every method on the crop grid with the same tissue maps
        fodf_volume = fodf(cfg, session, name, base, device, anat)
        streamlines = run_tracking(
            on_ref_grid(fodf_volume, anat), anat["affine"], anat["pve"], seeds, cfg, cfg.seed, 1, tracker
        )  # one thread: sessions run in parallel
        ends = endpoints(streamlines, anat, cfg)
        lower, higher, pair, keep = area_pairs(ends, anat)
        streamlines = [streamlines[index] for index in keep]
        out = dict(
            a=lower,
            b=higher,
            pair=pair,
            length=ends["length"][keep],
            n_seeds=len(seeds),
            n_streamlines=len(ends["v0"]),
            n_both_cortex=int(((ends["v0"] >= 0) & (ends["v1"] >= 0)).sum()),
            density=density_map(streamlines, anat["affine"], tuple(anat["shape"])).astype(np.float32),
            points=np.concatenate(streamlines) if streamlines else np.zeros((0, 3), np.float32),
            offsets=np.cumsum([0] + [len(streamline) for streamline in streamlines]),
        )
    out["settings"] = np.array(settings)
    atomic(path, lambda temporary: np.savez(temporary, **out))
    return out


def proximity_tract(anat, cfg, scale_mm=5.0, max_mm=30.0):
    """Diffusion-free lower bound: each seed vertex joined to a vertex of another area, p ~ exp(-distance / scale)."""
    rng = np.random.default_rng(cfg.seed)
    roi_vertices = np.flatnonzero(anat["area"] > 0)
    tree = cKDTree(anat["verts"][roi_vertices])
    seed_ends, partner_ends = [], []
    for vertex in roi_vertices.repeat(cfg.seeds_per_vertex):
        candidates = roi_vertices[tree.query_ball_point(anat["verts"][vertex], max_mm)]
        candidates = candidates[
            (anat["area"][candidates] != anat["area"][vertex]) & (anat["hemi"][candidates] == anat["hemi"][vertex])
        ]
        if len(candidates):
            weights = np.exp(-np.linalg.norm(anat["verts"][candidates] - anat["verts"][vertex], axis=1) / scale_mm)
            seed_ends.append(vertex)
            partner_ends.append(rng.choice(candidates, p=weights / weights.sum()))
    seed_ends, partner_ends = np.array(seed_ends, int), np.array(partner_ends, int)
    swap = anat["area"][seed_ends] > anat["area"][partner_ends]
    lower, higher = np.where(swap, partner_ends, seed_ends), np.where(swap, seed_ends, partner_ends)
    return dict(
        a=lower,
        b=higher,
        pair=anat["area"][lower] * 10 + anat["area"][higher],
        length=np.linalg.norm(anat["verts"][lower] - anat["verts"][higher], axis=1).astype(np.float32),
        n_seeds=len(roi_vertices) * cfg.seeds_per_vertex,
        n_streamlines=len(lower),
        n_both_cortex=len(lower),
        density=np.zeros(tuple(anat["shape"]), np.float32),
        points=np.zeros((0, 3), np.float32),
        offsets=np.zeros(1, int),
    )


# ---------------------------------------------------------------- scores
def scores(
    cfg, session, name, method, ref=("REF", "msmt_csd"), anat=None, graph=None, group=None, atlas=None, device="cpu"
):
    """One row of results for (session, protocol, method). group / atlas: alternative keys (group-average pRF maps,
    anatomical atlas); each adds its own RSI columns with the prefix template_ / atlas_."""
    cfg = cfg_for(cfg, session)
    # cache key: the tracking settings and the analysis settings that change the scores but not the tractogram
    signature = json.dumps(
        [
            dict(cfg.tracking),
            cfg.seeds_per_vertex,
            cfg.prf,
            cfg.n_null,
            cfg.seed,
            list(ref) if ref else None,
            graph is not None,
            group is not None,
            atlas is not None,
        ],
        sort_keys=True,
    )
    cache = (
        sdir(cfg, session, method) / f"scores_{name}_{method}_{hashlib.md5(signature.encode()).hexdigest()[:8]}.json"
    )
    if cache.exists():  # already scored with these analysis settings
        return load_json(cache)
    anat = anat if anat is not None else load_anat(cfg, session)
    tractogram = tract(cfg, session, name, method, device, anat)
    row = dict(
        sid=session["sid"],
        kind=session["kind"],
        protocol=name,
        method=method,
        n_seeds=int(tractogram["n_seeds"]),
        n_streamlines=int(tractogram["n_streamlines"]),
        n_both_cortex=int(tractogram["n_both_cortex"]),
        n_pairs=int(len(tractogram["a"])),
        length_median=float(np.median(tractogram["length"])) if len(tractogram["a"]) else np.nan,
    )
    if "prf_ang" in anat:
        row.update(
            M.retinotopic_scores(
                anat, tractogram["a"], tractogram["b"], tractogram["pair"], cfg, graph=graph, seed=cfg.seed
            )
        )
        row["noise_ceiling_deg"] = M.noise_ceiling(anat, cfg)
        for prefix, src in (("template", group), ("atlas", atlas)):
            if src is not None:
                alternative = M.retinotopic_scores(
                    anat,
                    tractogram["a"],
                    tractogram["b"],
                    tractogram["pair"],
                    cfg,
                    key=M.get_key(anat, cfg, group=src),
                    seed=cfg.seed,
                )
                row.update(
                    {
                        f"{prefix}_{key}": value
                        for key, value in alternative.items()
                        if key.startswith("rsi_ree_") and "geodesic" not in key
                    }
                )
                row[f"{prefix}_n_valid"] = alternative["n_valid"]
    if ref and session["kind"] == "3T" and method != "proximity" and (name, method) != tuple(ref):
        reference = tract(cfg, session, ref[0], ref[1], device, anat)
        row["wdice_ref"] = M.weighted_dice(tractogram["density"], reference["density"])
        row["badj_ref_mm"] = M.bundle_adjacency(tractogram["density"], reference["density"], cfg.ref_voxel)
        reference_fodf, predicted_fodf = fodf(cfg, session, ref[0], ref[1], device, anat), fodf(
            cfg, session, name, split_method(method)[0], device, anat
        )
        row["acc"], row["peak_err_deg"] = M.voxel_metrics(
            on_ref_grid(predicted_fodf, anat), reference_fodf["sh"], anat["pve"][0] > 0.5
        )
    save_json(row, cache)
    return row


def streamlines_of(tractogram):
    return [
        tractogram["points"][start:stop] for start, stop in zip(tractogram["offsets"][:-1], tractogram["offsets"][1:])
    ]


def retest_scores(cfg, sid, name="REF", method="msmt_csd", kinds=("3T", "retest")):
    """Agreement of the V1-V3 tractograms of two scans of one person. HCP retest is first aligned rigidly on the
    corresponding surface vertices; NSD runs (kinds=("nsd", "nsd2")) share one space."""
    first, second = session(sid, kinds[0]), session(sid, kinds[1])
    first_anat, second_anat = load_anat(cfg, first), load_anat(cfg, second)
    first_tract, second_tract = tract(cfg, first, name, method, anat=first_anat), tract(
        cfg, second, name, method, anat=second_anat
    )
    roi = first_anat["area"] > 0
    second_streamlines, rms = streamlines_of(second_tract), 0.0
    if kinds[0] == "3T":
        # Kabsch fit of retest to test on the V1-V3 vertices (same fsLR mesh), no reflection; rms in mm
        moving, fixed = second_anat["verts"][roi], first_anat["verts"][roi]
        moving_centre, fixed_centre = moving.mean(0), fixed.mean(0)
        left, _, right = np.linalg.svd((moving - moving_centre).T @ (fixed - fixed_centre))
        reflection = np.diag([1, 1, np.sign(np.linalg.det(right.T @ left.T))])
        rotation = right.T @ reflection @ left.T
        second_streamlines = [
            (streamline - moving_centre) @ rotation.T + fixed_centre for streamline in second_streamlines
        ]
        rms = float(np.sqrt((((moving - moving_centre) @ rotation.T + fixed_centre - fixed) ** 2).sum(1).mean()))
    second_density = density_map(second_streamlines, first_anat["affine"], tuple(first_anat["shape"]))
    first_counts = np.bincount(np.r_[first_tract["a"], first_tract["b"]], minlength=len(roi))[roi]
    second_counts = np.bincount(np.r_[second_tract["a"], second_tract["b"]], minlength=len(roi))[roi]
    return dict(
        sid=str(sid),
        protocol=name,
        method=method,
        wdice_retest=M.weighted_dice(first_tract["density"], second_density),
        badj_retest_mm=M.bundle_adjacency(first_tract["density"], second_density, cfg.ref_voxel),
        align_rms_mm=rms,
        vertex_count_r=(
            float(np.corrcoef(first_counts, second_counts)[0, 1])
            if first_counts.std() > 0 and second_counts.std() > 0
            else np.nan
        ),
    )


# ---------------------------------------------------------------- subject lists and result tables
def make_subject_lists(cfg, s3=None):
    """Seeded subject lists, saved to outputs/subjects.json. Test subjects have measured 7T retinotopy; training and
    validation subjects come from the rest of HCP. development_subjects are in no test list (test, t7, retest)."""
    path = cfg.P.out / "subjects.json"
    development = {str(sid) for sid in cfg.development_subjects or []}
    if path.exists():
        lists = load_json(path)
        for group in ("test", "train", "val"):
            wanted = cfg.get(f"n_{group}")
            if not cfg.subjects and wanted is not None and wanted >= 0 and len(lists[group]) != wanted:
                raise SystemExit(
                    f"{path} holds {len(lists[group])} {group} subjects and config.yaml asks for n_{group}: "
                    f"{wanted}. Set paths.out to a new folder for a run of a different size, or delete the "
                    "file to draw again."
                )
        if not cfg.subjects and development & set(lists["test"] + lists.get("retest", [])):
            raise SystemExit(f"{path} has development subjects in a test list; delete the file to draw again.")
        return lists
    if cfg.subjects:  # DEV_MODE: the few subjects named in config.yaml
        lists = {group: list(sids) for group, sids in cfg.subjects.items()}
    else:
        s3 = s3 or hcp_client(cfg)
        rng = np.random.default_rng(cfg.seed)
        has_scan = lambda release, sid, folder: s3_exists(s3, HCP_BUCKET, f"{release}/{sid}/T1w/{folder}/data.nii.gz")
        take = lambda candidates, count, usable: list(
            itertools.islice((sid for sid in candidates if usable(sid)), None if count < 0 else count)
        )
        with_retinotopy = [sid for sid in prf_subject_ids(cfg) if not sid.startswith("9999")]  # 9999xx: group averages
        test = take(
            list(rng.permutation(with_retinotopy)),
            cfg.n_test,
            lambda sid: sid not in development and has_scan("HCP_1200", sid, "Diffusion"),
        )
        retest_all = s3_dirs(s3, HCP_BUCKET, "HCP_Retest/")
        pool = sorted(set(s3_dirs(s3, HCP_BUCKET, "HCP_1200/")) - set(with_retinotopy) - set(retest_all))
        assert cfg.n_val >= 0, "n_val must be a number: validation subjects are set aside from the training pool"
        trainval = take(
            list(rng.permutation(pool)),
            -1 if cfg.n_train < 0 else cfg.n_val + cfg.n_train,
            lambda sid: has_scan("HCP_1200", sid, "Diffusion"),
        )
        # validation first, so it does not depend on n_train
        n_nsd = 8 if cfg.n_nsd < 0 else cfg.n_nsd
        lists = dict(
            test=test,
            train=trainval[cfg.n_val :],
            val=trainval[: cfg.n_val],
            nsd=[f"subj{number:02d}" for number in range(1, n_nsd + 1)],
            t7=take(test, cfg.n_7t, lambda sid: has_scan("HCP_1200", sid, "Diffusion_7T")),
            retest=take(
                sorted(set(retest_all) & set(with_retinotopy)),
                cfg.n_retest,
                lambda sid: sid not in development and has_scan("HCP_Retest", sid, "Diffusion"),
            ),
        )
        for group in ("test", "train", "val"):
            wanted = cfg[f"n_{group}"]
            assert (
                wanted < 0 or len(lists[group]) == wanted
            ), f"only {len(lists[group])} {group} subjects found, n_{group} is {wanted}"
        print("subjects:", {group: len(sids) for group, sids in lists.items()}, flush=True)
    save_json(lists, path)
    return lists


def robustness_rows(cfg, ses, variants):
    """Reference RSI of one session under other settings (`variants`: name -> {"section.key": value}), each pRF
    half, and dorsal / ventral / left / right subsets. tracking.* changes are tracked again from the same fODF."""
    anat = load_anat(cfg, ses)
    reference = tract(cfg, ses, "REF", "msmt_csd", anat=anat)
    score = lambda adjusted, tractogram=reference, selected=slice(None), **options: M.retinotopic_scores(
        anat,
        tractogram["a"][selected],
        tractogram["b"][selected],
        tractogram["pair"][selected],
        adjusted,
        seed=cfg.seed,
        **options,
    )
    rows = []
    for name, changes in variants.items():
        adjusted = copy.deepcopy(cfg)
        for setting, value in changes.items():
            section, key = setting.split(".")
            adjusted[section][key] = value
        tracking_changes = {key: value for key, value in changes.items() if key.startswith("tracking.")}
        if tracking_changes:
            tag = "".join(f"_{key.split('.')[1]}{value}" for key, value in tracking_changes.items())
            rows.append(
                dict(
                    sid=ses["sid"],
                    analysis=name,
                    **score(adjusted, tract(adjusted, ses, "REF", "msmt_csd", anat=anat, tag=tag)),
                )
            )
        else:
            rows.append(dict(sid=ses["sid"], analysis=name, **score(adjusted)))
    for fit in (1, 2):
        rows.append(dict(sid=ses["sid"], analysis=f"pRF half {fit}", **score(cfg, key=M.get_key(anat, cfg, fit))))
    for subset, selected in (
        ("dorsal", anat["dorsal"][reference["a"]]),
        ("ventral", ~anat["dorsal"][reference["a"]]),
        ("left", anat["hemi"][reference["a"]] == 0),
        ("right", anat["hemi"][reference["a"]] == 1),
    ):
        if selected.sum() > 50:
            rows.append(dict(sid=ses["sid"], analysis=subset, **score(cfg, selected=selected)))
    return rows


def _gpu_fits(cfg, sessions, fits, device, errors):
    try:
        for session in sessions:
            for name, method in fits:
                if not (sdir(cfg, session, method) / f"fodf_{name}_{method}.npz").exists():
                    fodf(cfg, session, name, method, device)
    except BaseException as error:
        errors.append(error)


def _grid_rows(cfg, session, combos, options, last=()):
    anat = load_anat(cfg, session)
    keyed = "prf_ang" in anat
    graph = M.Geodesic(anat) if keyed else None  # surface distances, shared by all combinations
    group = group_key_source(cfg, anat) if keyed else None
    atlas = atlas_key_source(cfg, anat, session) if keyed else None
    order = sorted(range(len(combos)), key=lambda index: (combos[index][0], split_method(combos[index][1])[0]) in last)
    rows = {
        index: scores(cfg, session, *combos[index], anat=anat, graph=graph, group=group, atlas=atlas, **options)
        for index in order
    }
    return [rows[index] for index in range(len(combos))]


def run_grid(cfg, sessions, combos, name, device="cpu", **options):
    """Scores for every session x (protocol, method), saved to outputs/results/<name>.csv. One session per worker;
    with a CUDA device a thread also fits the GPU-solvable fODFs, from the last session backwards."""
    import pandas as pd, threading

    fits = [
        fit
        for fit in dict.fromkeys((name, split_method(method)[0]) for name, method in combos)
        if fit[1] != "proximity" and not fit[1].startswith("net_") and _gpu_fit(cfg, *fit, device)
    ]
    for hemi in cfg.hemis:  # atlas files: fetched here, not by every worker
        atlas_files(cfg, hemi)
        if any(not session["kind"].startswith("nsd") for session in sessions):
            nsd.bridge_file(cfg, hemi)
    errors = []
    thread = threading.Thread(target=_gpu_fits, args=(cfg, sessions[::-1], fits, device, errors), daemon=True)
    if fits:
        thread.start()
    rows = each(
        cfg, lambda worker_cfg, session: _grid_rows(worker_cfg, session, combos, options, set(fits)), sessions, name
    )
    if fits:
        thread.join()
    if errors:
        raise errors[0]
    df = pd.DataFrame([row for session_rows in rows for row in session_rows])
    folder = cfg.P.out / "results"
    folder.mkdir(exist_ok=True)
    df.to_csv(folder / f"{name}.csv", index=False)
    return df


# ---------------------------------------------------------------- download everything
def download_all(cfg, workers=16, list_only=False, connections=256):
    """Fetch every file the configured run needs (OSF, HCP S3, NSD); complete files are skipped.
    connections: 8 MB parts fetched at once; workers: parallel size checks. Returns (key, bytes, etag) rows."""
    osf_download(cfg)
    subjects = make_subject_lists(cfg)
    print({key: len(entry) for key, entry in subjects.items()}, flush=True)
    jobs = [
        ("hcp", sid, "3T", "HCP_1200")
        for sid in subjects["test"] + subjects["train"] + subjects["val"] + subjects["retest"]
    ]
    jobs += [("hcp", sid, "7T", "HCP_1200") for sid in subjects["t7"]] + [
        ("hcp", sid, "3T", "HCP_Retest") for sid in subjects["retest"]
    ]
    jobs += [
        ("nsd", sid, run, None) for sid in subjects.get("nsd", []) for run in ((1, 2) if cfg.nsd.use_run2 else (1,))
    ]
    cached = lambda job: (
        (cfg.P.cache / session(job[1], "retest" if job[3] == "HCP_Retest" else job[2])["tag"] / "dwi.npz").exists()
        if job[0] == "hcp"
        else (cfg.P.cache / f"{job[1]}_{'nsd2' if job[2] == 2 else 'nsd'}" / "dwi.npz").exists()
    )
    jobs = [job for job in dict.fromkeys(jobs) if not cached(job)]  # already cropped: raw files not needed
    print(len(jobs), "scans to fetch or verify", flush=True)
    if list_only:
        return []
    s3, pub = hcp_client(cfg), anon_client()
    remove_stale_parts(cfg.P.hcp)
    remove_stale_parts(cfg.P.nsd)
    for hemi in "LR":
        nsd.bridge_file(cfg, hemi)
        atlas_files(cfg, hemi)
    files = [
        (s3, HCP_BUCKET, key, cfg.P.hcp / key)
        for job in jobs
        if job[0] == "hcp"
        for key in hcp_keys(job[1], job[2], job[3])
    ]
    files += [
        (pub, nsd.BUCKET, key, cfg.P.nsd / key)
        for job in jobs
        if job[0] == "nsd"
        for key in nsd.nsd_keys(job[1], job[2])
    ]
    # inflated surfaces of the test subjects for the figures (notebook 10); fetched even when a scan is cropped
    files += [(s3, HCP_BUCKET, key, cfg.P.hcp / key) for sid in subjects["test"] for key in hcp_figure_keys(sid)]
    files = list({file[2]: file for file in files}.values())
    rows = _fetch_files(files, workers, connections)
    manifest = update_manifest(cfg, rows)
    print(
        len(manifest),
        "files in data/manifest.json,",
        round(sum(entry["bytes"] for entry in manifest.values()) / 1e9, 1),
        "GB",
        flush=True,
    )
    return rows


def _fetch_files(files, workers, connections):
    """Download the missing (client, bucket, key, dest) files with one progress bar. Returns (key, bytes, etag)."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import tqdm

    with ThreadPoolExecutor(workers) as executor:
        sizes = list(
            tqdm(executor.map(lambda file: s3_size(*file), files), total=len(files), desc="checking files on disk")
        )
    total, todo = sum(size[0] for size in sizes), sum(size[1] for size in sizes)
    print(
        f"{len(files)} files, {total / 1e9:.1f} GB: {sum(size[1] > 0 for size in sizes)} files, {todo / 1e9:.1f} GB "
        f"to download",
        flush=True,
    )
    need = [(*file, size[0], size[2]) for file, size in zip(files, sizes) if size[1]]
    lock = threading.Lock()
    done = [0]
    with tqdm(
        total=todo,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="whole dataset",
        smoothing=0.05,
        postfix=f"0/{len(need)} files",
    ) as bar:

        def progress(count):
            with lock:
                bar.update(count)

        def finished(row):
            with lock:
                done[0] += 1
                bar.set_postfix_str(f"{done[0]}/{len(need)} files")

        rows = s3_get_many(need, connections, progress, finished)
    return rows + [(file[2], size[0], size[2]) for file, size in zip(files, sizes) if not size[1]]


if __name__ == "__main__":  # python -m retinopath.pipeline [--dev] [--connections N] [--list]
    import argparse
    from .core import load_config

    parser = argparse.ArgumentParser(description="Download all data needed by config.yaml")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--list", action="store_true", help="only build the subject lists and count the scans")
    parser.add_argument("--connections", type=int, default=256, help="8 MB parts fetched at once, over all files")
    arguments = parser.parse_args()
    import resource  # each connection is an open file; Linux allows 1024 by default

    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard if hard != resource.RLIM_INFINITY else 65536, hard))
    download_all(load_config(arguments.dev), arguments.workers, arguments.list, min(arguments.connections, 4096))
