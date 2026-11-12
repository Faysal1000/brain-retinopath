"""Per-session caching: crop -> protocol -> fODF -> tractogram -> scores. Every step skips work already on disk.

A "session" is one scan of one subject: kind 3T (main), 7T (same subject space, 7T diffusion), retest, or
nsd / nsd2 (first / second NSD diffusion run). Cache files live in cfg.P.cache/<tag>/.

A "method" names the fODF and, after a plus sign, the tracker: csd, rumba, net_path_s0, csd+prob. No suffix means
particle-filter tracking.
"""
import copy, hashlib, json, shutil, threading
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from . import metrics as M, nsd
from .atlas import atlas_files, atlas_key_source
from .core import atomic, load_json, n_jobs, save_json
from .degrade import make_protocol, to_grid
from .download import HCP_BUCKET, anon_client, hcp_client, hcp_download, hcp_figure_keys, hcp_keys, osf_download, s3_dirs, s3_exists, s3_get, s3_size, s3_get_many, update_manifest, remove_stale_parts
from .recon import fit_fodf, signal_sh
from .subject import build_anat, load_dwi_crop, load_prf, prf_subject_ids, subject_dir
from .track import area_pairs, density_map, endpoints, make_seeds, run_tracking

CLASSICAL = {"csd": "msmt_csd", "rumba": "rumba"}       # label -> recon method ("csd" is multi-shell when it can be)


def split_method(method):
    """'csd+prob' -> ('csd', 'prob'); 'csd' -> ('csd', 'pft')."""
    base, _, tracker = method.partition("+")
    return base, tracker or "pft"


def session(sid, kind="3T"):
    """dict(sid, kind, release, dwi folder name, tag = cache folder name: sid for 3T, else sid_kind)."""
    return dict(sid=str(sid), kind=kind, release="HCP_Retest" if kind == "retest" else "HCP_1200",
                dwi="Diffusion_7T" if kind == "7T" else "Diffusion", tag=str(sid) if kind == "3T" else f"{sid}_{kind}")


def cfg_for(cfg, ses):
    """Settings for this session: NSD has its own seed density and eccentricity limit (config.yaml, `nsd:`)."""
    if not ses["kind"].startswith("nsd"):
        return cfg
    c = copy.deepcopy(cfg); c["seeds_per_vertex"] = cfg.nsd.seeds_per_vertex; c.prf["ecc_max"] = cfg.nsd.ecc_max
    return c


def sdir(cfg, ses):
    d = cfg.P.cache / ses["tag"]; d.mkdir(parents=True, exist_ok=True); return d


# ---------------------------------------------------------------- crop
def prepare(cfg, ses, s3=None):
    """Download (if needed) and cache the occipital crop of one session."""
    d = sdir(cfg, ses)
    if (d / "anat.npz").exists() and (d / "dwi.npz").exists():
        return d
    if ses["kind"].startswith("nsd"):                            # nsd = first diffusion run, nsd2 = second run
        run = 2 if ses["kind"] == "nsd2" else 1
        nsd.nsd_download(cfg, ses["sid"], run)
        anat = nsd.build_anat(cfg, ses["sid"], run)
        dwi, bvals, bvecs, mask = nsd.load_dwi_crop(cfg, ses["sid"], anat, run)
        atomic(d / "dwi.npz", lambda f: np.savez(f, dwi=dwi, bvals=bvals, bvecs=bvecs, mask=mask)); atomic(d / "anat.npz", lambda f: np.savez(f, **anat))
        return d
    raw = subject_dir(cfg, ses["sid"], ses["release"])
    if not (raw / ses["dwi"] / "data.nii.gz").exists() or not (raw / "aparc+aseg.nii.gz").exists():
        if ses["kind"] != "7T" or not (raw / "aparc+aseg.nii.gz").exists():
            hcp_download(cfg, ses["sid"], "3T", ses["release"], s3)      # anatomy + 3T diffusion
        if ses["kind"] == "7T":
            hcp_download(cfg, ses["sid"], "7T", ses["release"], s3)
    anat = build_anat(cfg, ses["sid"], ses["release"], ses["dwi"])
    dwi, bvals, bvecs, mask = load_dwi_crop(cfg, ses["sid"], anat, ses["release"], ses["dwi"])
    atomic(d / "dwi.npz", lambda f: np.savez(f, dwi=dwi, bvals=bvals, bvecs=bvecs, mask=mask))
    atomic(d / "anat.npz", lambda f: np.savez(f, **anat))
    if not cfg.keep_raw:
        shutil.rmtree(raw / ses["dwi"], ignore_errors=True)
    return d


def load_anat(cfg, ses):
    a = dict(np.load(sdir(cfg, ses) / "anat.npz"))
    a["sid"], a["hemis"] = str(a["sid"]), str(a["hemis"])
    return a


def load_dwi(cfg, ses):
    z = np.load(sdir(cfg, ses) / "dwi.npz")
    return z["dwi"], z["bvals"], z["bvecs"], z["mask"]


def group_key_source(cfg, anat, sid="999999"):
    """Group-average retinotopy placed on this subject's vertices: an atlas-style ('template') key."""
    if "tpl_ang" in anat:                                        # NSD: carried over through the fsaverage sphere
        return {f"prf_{q}": anat[f"tpl_{q}"] for q in ("ang", "ecc", "r2")}
    prf = load_prf(cfg, sid)
    return {f"prf_{q}": np.concatenate([prf[h][q] for h in anat["hemis"]]) for q in ("ang", "ecc", "r2")}


# ---------------------------------------------------------------- protocol
def protocol(cfg, ses, name, anat=None):
    """Simulated acquisition `name` (config.yaml) or the scan as acquired (name == 'NATIVE')."""
    anat = anat if anat is not None else load_anat(cfg, ses)
    dwi, bvals, bvecs, _ = load_dwi(cfg, ses)
    if name == "NATIVE":
        return dict(dwi=dwi, bvals=bvals, bvecs=bvecs, affine=anat["affine"], pve=anat["pve"])
    return make_protocol(dwi, bvals, bvecs, anat["affine"], anat["pve"], cfg.protocols[name], cfg.n_b0,
                         seed=cfg.seed + int(ses["sid"]) % 100000)


def anat_channels(anat):
    """7 anatomical input channels: wm, gm, csf, signed distance / 5 mm, inward normal (3)."""
    return np.concatenate([anat["pve"], np.clip(anat["sdist"] / 5.0, -2, 2)[None], anat["normal_field"]]).astype(np.float32)


def net_input(cfg, ses, name, anat=None, source="signal"):
    """Network input for protocol `name` on the crop grid, cached as float16.
    source = signal: per-shell SH of the diffusion signal. source = fodf: the classical CSD fODF of that protocol
    (input of the fODF-to-fODF baseline)."""
    f = sdir(cfg, ses) / (f"in_{name}.npy" if source == "signal" else f"in_{name}_fodf.npy")
    if f.exists():
        return np.load(f)
    anat = anat if anat is not None else load_anat(cfg, ses)
    if source == "fodf":
        x = on_ref_grid(fodf(cfg, ses, name, "csd", anat=anat), anat).astype(np.float16)
    else:
        p = protocol(cfg, ses, name, anat)
        x = signal_sh(p["dwi"], p["bvals"], p["bvecs"], cfg.shells_in, cfg.sh_order_in)
        x = np.moveaxis(to_grid(np.moveaxis(x, 0, -1), anat["shape"]), -1, 0).astype(np.float16)
    atomic(f, lambda t: np.save(t, x)); return x


# ---------------------------------------------------------------- fODF
def fodf(cfg, ses, name, method, device="cpu", anat=None):
    """fODF of `method` on protocol `name`. Returns dict(sh, affine, pve). Methods: msmt_csd (reference), csd, rumba,
    csd_mppca (denoise first), csd_interp (interpolate the DWIs to the reference grid first)."""
    f = sdir(cfg, ses) / f"fodf_{name}_{method}.npz"
    if f.exists():
        z = np.load(f); return dict(sh=z["sh"].astype(np.float32), affine=z["affine"], pve=z["pve"])
    if method.startswith("net_"):
        raise FileNotFoundError(f"{f.name}: network outputs are written by notebook 06 (train.net_fodf)")
    anat = anat if anat is not None else load_anat(cfg, ses)
    p = protocol(cfg, ses, name, anat)
    if method == "csd_mppca":
        from dipy.denoise.localpca import mppca
        p["dwi"] = mppca(p["dwi"], patch_radius=2).astype(np.float32)
    if method == "csd_interp":
        p.update(dwi=to_grid(p["dwi"], anat["shape"]), affine=anat["affine"], pve=anat["pve"])
    recon = cfg.ref_model if method == "msmt_csd" else CLASSICAL.get(method.split("_")[0], "msmt_csd")
    sh = fit_fodf(p, p["pve"][0] > cfg.wm_mask_thr, recon, cfg, n_jobs(cfg), device)
    return save_fodf(cfg, ses, name, method, sh, p["affine"], p["pve"])


def save_fodf(cfg, ses, name, method, sh, affine, pve):
    atomic(sdir(cfg, ses) / f"fodf_{name}_{method}.npz", lambda f: np.savez(f, sh=sh.astype(np.float16), affine=affine, pve=pve.astype(np.float16)))
    return dict(sh=sh, affine=affine, pve=pve)


def on_ref_grid(fo, anat):
    """SH volume resampled to the crop grid (for voxel-wise comparison with the reference)."""
    if tuple(fo["sh"].shape[1:]) == tuple(anat["shape"]):
        return fo["sh"]
    return np.moveaxis(to_grid(np.moveaxis(fo["sh"], 0, -1), anat["shape"]), -1, 0)


# ---------------------------------------------------------------- tractogram
def tract(cfg, ses, name, method, device="cpu", anat=None):
    """Tractogram reduced to what the analysis needs: end vertices of V1-V3 streamlines, counts, density map,
    and the streamlines themselves (for figures and test-retest overlap). Same seeds for every method."""
    f = sdir(cfg, ses) / f"tract_{name}_{method}.npz"
    if f.exists():
        return dict(np.load(f))
    anat = anat if anat is not None else load_anat(cfg, ses)
    cfg = cfg_for(cfg, ses)
    seeds, _ = make_seeds(anat, cfg, cfg.seed)
    if method == "proximity":
        out = proximity_tract(anat, cfg)
    else:
        base, tracker = split_method(method)
        # Every method is tracked on the crop grid with the same tissue maps (fODFs of coarser protocols are
        # interpolated), so differences between methods come from the fODF and not from the stopping rule.
        fo = fodf(cfg, ses, name, base, device, anat)
        sl = run_tracking(on_ref_grid(fo, anat), anat["affine"], anat["pve"], seeds, cfg, cfg.seed, n_jobs(cfg), tracker)
        ep = endpoints(sl, anat, cfg)
        a, b, pair, keep = area_pairs(ep, anat)
        sl = [sl[i] for i in keep]
        out = dict(a=a, b=b, pair=pair, length=ep["length"][keep], n_seeds=len(seeds), n_streamlines=len(ep["v0"]),
                   n_both_cortex=int(((ep["v0"] >= 0) & (ep["v1"] >= 0)).sum()),
                   density=density_map(sl, anat["affine"], tuple(anat["shape"])).astype(np.float32),
                   points=np.concatenate(sl) if sl else np.zeros((0, 3), np.float32),
                   offsets=np.cumsum([0] + [len(s) for s in sl]))
    atomic(f, lambda t: np.savez(t, **out)); return out


def proximity_tract(anat, cfg, scale_mm=5.0, max_mm=30.0):
    """Lower bound that knows nothing about diffusion: every seed vertex is joined to a vertex of another area in the
    same hemisphere, chosen with probability falling off with straight-line distance."""
    rng = np.random.default_rng(cfg.seed)
    roi = np.flatnonzero(anat["area"] > 0); tree = cKDTree(anat["verts"][roi]); A, B = [], []
    for v in roi.repeat(cfg.seeds_per_vertex):
        c = roi[tree.query_ball_point(anat["verts"][v], max_mm)]
        c = c[(anat["area"][c] != anat["area"][v]) & (anat["hemi"][c] == anat["hemi"][v])]
        if len(c):
            w = np.exp(-np.linalg.norm(anat["verts"][c] - anat["verts"][v], axis=1) / scale_mm)
            A.append(v); B.append(rng.choice(c, p=w / w.sum()))
    A, B = np.array(A, int), np.array(B, int); swap = anat["area"][A] > anat["area"][B]
    a, b = np.where(swap, B, A), np.where(swap, A, B)
    return dict(a=a, b=b, pair=anat["area"][a] * 10 + anat["area"][b],
                length=np.linalg.norm(anat["verts"][a] - anat["verts"][b], axis=1).astype(np.float32),
                n_seeds=len(roi) * cfg.seeds_per_vertex, n_streamlines=len(a), n_both_cortex=len(a),
                density=np.zeros(tuple(anat["shape"]), np.float32), points=np.zeros((0, 3), np.float32), offsets=np.zeros(1, int))


# ---------------------------------------------------------------- scores
def scores(cfg, ses, name, method, ref=("REF", "msmt_csd"), anat=None, graph=None, group=None, atlas=None, device="cpu"):
    """One row of results for (session, protocol, method). group / atlas: alternative keys (group-average pRF maps,
    anatomical atlas); each adds its own RSI columns with the prefix template_ / atlas_."""
    cfg = cfg_for(cfg, ses)
    # cache key: analysis settings that change the scores but not the tractogram
    sig = json.dumps([cfg.prf, cfg.n_null, cfg.seed, list(ref) if ref else None, graph is not None, group is not None,
                      atlas is not None], sort_keys=True)
    cache = sdir(cfg, ses) / f"scores_{name}_{method}_{hashlib.md5(sig.encode()).hexdigest()[:8]}.json"
    if cache.exists():                                           # already scored with these analysis settings
        return load_json(cache)
    anat = anat if anat is not None else load_anat(cfg, ses)
    t = tract(cfg, ses, name, method, device, anat)
    row = dict(sid=ses["sid"], kind=ses["kind"], protocol=name, method=method, n_seeds=int(t["n_seeds"]),
               n_streamlines=int(t["n_streamlines"]), n_both_cortex=int(t["n_both_cortex"]), n_pairs=int(len(t["a"])),
               length_median=float(np.median(t["length"])) if len(t["a"]) else np.nan)
    if "prf_ang" in anat:
        row.update(M.retinotopic_scores(anat, t["a"], t["b"], t["pair"], cfg, graph=graph, seed=cfg.seed))
        row["noise_ceiling_deg"] = M.noise_ceiling(anat, cfg)
        for prefix, src in (("template", group), ("atlas", atlas)):
            if src is not None:
                g = M.retinotopic_scores(anat, t["a"], t["b"], t["pair"], cfg, key=M.get_key(anat, cfg, group=src), seed=cfg.seed)
                row.update({f"{prefix}_{k}": v for k, v in g.items() if k.startswith("rsi_ree_") and "geodesic" not in k})
                row[f"{prefix}_n_valid"] = g["n_valid"]
    if ref and ses["kind"] == "3T" and method != "proximity" and (name, method) != tuple(ref):
        r = tract(cfg, ses, ref[0], ref[1], device, anat)
        row["wdice_ref"] = M.weighted_dice(t["density"], r["density"])
        row["badj_ref_mm"] = M.bundle_adjacency(t["density"], r["density"], cfg.ref_voxel)
        fr, fp = fodf(cfg, ses, ref[0], ref[1], device, anat), fodf(cfg, ses, name, split_method(method)[0], device, anat)
        row["acc"], row["peak_err_deg"] = M.voxel_metrics(on_ref_grid(fp, anat), fr["sh"], anat["pve"][0] > 0.5)
    save_json(row, cache); return row


def streamlines_of(t):
    return [t["points"][i:j] for i, j in zip(t["offsets"][:-1], t["offsets"][1:])]


def retest_scores(cfg, sid, name="REF", method="msmt_csd", kinds=("3T", "retest")):
    """Agreement of the V1-V3 tractogram between two scans of one person: bundle overlap and correlation of
    per-vertex streamline-end counts. HCP retest has its own anatomy, so the second scan is first aligned rigidly
    using corresponding surface vertices; the two NSD runs already share one space (kinds=("nsd", "nsd2"))."""
    s1, s2 = session(sid, kinds[0]), session(sid, kinds[1])
    a1, a2 = load_anat(cfg, s1), load_anat(cfg, s2)
    t1, t2 = tract(cfg, s1, name, method, anat=a1), tract(cfg, s2, name, method, anat=a2)
    roi = a1["area"] > 0; sl2, rms = streamlines_of(t2), 0.0
    if kinds[0] == "3T":
        # Rigid retest -> test alignment (Kabsch). Both scans use the same fsLR 32k mesh, so vertex i is the same
        # cortical point in both; V1-V3 vertices serve as point pairs. D prevents a reflection.
        # rms = residual vertex distance (mm) after alignment.
        P, Q = a2["verts"][roi], a1["verts"][roi]; pc, qc = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc)); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        sl2 = [(s - pc) @ R.T + qc for s in sl2]; rms = float(np.sqrt((((P - pc) @ R.T + qc - Q) ** 2).sum(1).mean()))
    d2 = density_map(sl2, a1["affine"], tuple(a1["shape"]))
    c1 = np.bincount(np.r_[t1["a"], t1["b"]], minlength=len(roi))[roi]; c2 = np.bincount(np.r_[t2["a"], t2["b"]], minlength=len(roi))[roi]
    return dict(sid=str(sid), protocol=name, method=method, wdice_retest=M.weighted_dice(t1["density"], d2),
                badj_retest_mm=M.bundle_adjacency(t1["density"], d2, cfg.ref_voxel), align_rms_mm=rms,
                vertex_count_r=float(np.corrcoef(c1, c2)[0, 1]) if c1.std() > 0 and c2.std() > 0 else np.nan)


# ---------------------------------------------------------------- subject lists and result tables
def make_subject_lists(cfg, s3=None):
    """Fixed, seeded subject lists saved to outputs/subjects.json. Test = subjects with measured 7T retinotopy.
    Training and validation subjects come from the rest of HCP, so no test subject is ever seen in training."""
    f = cfg.P.out / "subjects.json"
    if f.exists():
        return load_json(f)
    if cfg.subjects:                                             # DEV_MODE: the few subjects named in config.yaml
        lists = {k: list(v) for k, v in cfg.subjects.items()}
    else:
        s3 = s3 or hcp_client(cfg); rng = np.random.default_rng(cfg.seed)
        has = lambda rel, sid, dif: s3_exists(s3, HCP_BUCKET, f"{rel}/{sid}/T1w/{dif}/data.nii.gz")
        take = lambda cand, n, ok: [s for s, _ in zip((s for s in cand if ok(s)), range(n))]
        prf = [s for s in prf_subject_ids(cfg) if not s.startswith("9999")]          # 9999xx are group averages
        test = take(list(rng.permutation(prf)), cfg.n_test, lambda s: has("HCP_1200", s, "Diffusion"))
        retest_all = s3_dirs(s3, HCP_BUCKET, "HCP_Retest/")
        rest = sorted(set(s3_dirs(s3, HCP_BUCKET, "HCP_1200/")) - set(prf) - set(retest_all))
        tv = take(list(rng.permutation(rest)), cfg.n_train + cfg.n_val, lambda s: has("HCP_1200", s, "Diffusion"))
        lists = dict(test=test, train=tv[:cfg.n_train], val=tv[cfg.n_train:], nsd=[f"subj{i:02d}" for i in range(1, cfg.n_nsd + 1)],
                     t7=take(test, cfg.n_7t, lambda s: has("HCP_1200", s, "Diffusion_7T")),
                     retest=take(sorted(set(retest_all) & set(prf)), cfg.n_retest, lambda s: has("HCP_Retest", s, "Diffusion")))
        for k in ("test", "train", "val"):
            assert len(lists[k]) == cfg[f"n_{k}"], f"only {len(lists[k])} {k} subjects found, n_{k} is {cfg[f'n_{k}']}"
    save_json(lists, f); return lists


def _grid_rows(cfg, ses, combos, kw):
    anat = load_anat(cfg, ses)
    keyed = "prf_ang" in anat
    graph = True if keyed else None                              # True: built only when a score is not cached
    group = group_key_source(cfg, anat) if keyed else None
    atlas = atlas_key_source(cfg, anat, ses) if keyed else None
    return [scores(cfg, ses, p, m, anat=anat, graph=graph, group=group, atlas=atlas, **kw) for p, m in combos]


def run_grid(cfg, sessions, combos, name, device="cpu", **kw):
    """Scores for every session x (protocol, method); writes outputs/results/<name>.csv and returns the table.
    Missing fODFs are fitted first, in this process (the fits use all cores or the GPU themselves). Tracking and
    scoring then run one session per worker process, because the tracker is single-threaded. Everything is cached,
    so a second call only reads the scores."""
    import pandas as pd
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    for ses in tqdm(sessions, desc=f"{name}: fODF"):
        for proto, method in combos:
            base = split_method(method)[0]
            if base != "proximity" and not (sdir(cfg, ses) / f"fodf_{proto}_{base}.npz").exists():
                fodf(cfg, ses, proto, base, device)
    one = copy.deepcopy(cfg); one["n_jobs"] = 1
    rows = Parallel(n_jobs=min(n_jobs(cfg), len(sessions)))(delayed(_grid_rows)(one, s, combos, kw) for s in tqdm(sessions, desc=name))
    df = pd.DataFrame([r for rs in rows for r in rs]); out = cfg.P.out / "results"; out.mkdir(exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False); return df


# ---------------------------------------------------------------- download everything
def download_all(cfg, workers=16, list_only=False, connections=256):
    """Fetch every file the configured run needs: retinotopy results (OSF), HCP 3T / 7T / retest scans (S3, keys in
    .env) and NSD (public S3). Complete files are skipped, so it can be stopped and run again. connections: 8 MB
    parts fetched at once, over all files; workers: parallel size checks.
    Returns the list of (key, bytes, etag) rows; with list_only=True nothing large is downloaded."""
    osf_download(cfg)
    subjects = make_subject_lists(cfg)
    print({k: len(v) for k, v in subjects.items()}, flush=True)
    jobs = [("hcp", s, "3T", "HCP_1200") for s in subjects["test"] + subjects["train"] + subjects["val"] + subjects["retest"]]
    jobs += [("hcp", s, "7T", "HCP_1200") for s in subjects["t7"]] + [("hcp", s, "3T", "HCP_Retest") for s in subjects["retest"]]
    jobs += [("nsd", s, r, None) for s in subjects.get("nsd", []) for r in ((1, 2) if cfg.nsd.use_run2 else (1,))]
    cached = lambda j: (cfg.P.cache / session(j[1], "retest" if j[3] == "HCP_Retest" else j[2])["tag"] / "dwi.npz").exists() \
        if j[0] == "hcp" else (cfg.P.cache / f"{j[1]}_{'nsd2' if j[2] == 2 else 'nsd'}" / "dwi.npz").exists()
    jobs = [j for j in dict.fromkeys(jobs) if not cached(j)]                 # already cropped: raw files not needed
    print(len(jobs), "scans to fetch or verify", flush=True)
    if list_only:
        return []
    s3, pub = hcp_client(cfg), anon_client()
    remove_stale_parts(cfg.P.hcp); remove_stale_parts(cfg.P.nsd)
    for h in "LR":
        nsd.bridge_file(cfg, h); atlas_files(cfg, h)
    files = [(s3, HCP_BUCKET, k, cfg.P.hcp / k) for j in jobs if j[0] == "hcp" for k in hcp_keys(j[1], j[2], j[3])]
    files += [(pub, nsd.BUCKET, k, cfg.P.nsd / k) for j in jobs if j[0] == "nsd" for k in nsd.nsd_keys(j[1], j[2])]
    # inflated surfaces of the test subjects for the figures (notebook 10); fetched even when a scan is cropped
    files += [(s3, HCP_BUCKET, k, cfg.P.hcp / k) for s in subjects["test"] for k in hcp_figure_keys(s)]
    files = list({f[2]: f for f in files}.values())
    rows = _fetch_files(files, workers, connections)
    man = update_manifest(cfg, rows)
    print(len(man), "files in data/manifest.json,", round(sum(v["bytes"] for v in man.values()) / 1e9, 1), "GB", flush=True)
    return rows


def _fetch_files(files, workers, connections):
    """Download the files (client, bucket, key, dest) not yet on disk through `connections` shared connections, with
    one progress bar for all of them. Returns (key, bytes, etag) rows of the files on disk."""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import tqdm
    with ThreadPoolExecutor(workers) as ex:
        sizes = list(tqdm(ex.map(lambda f: s3_size(*f), files), total=len(files), desc="checking files on disk"))
    total, todo = sum(s[0] for s in sizes), sum(s[1] for s in sizes)
    print(f"{len(files)} files, {total / 1e9:.1f} GB: {sum(s[1] > 0 for s in sizes)} files, {todo / 1e9:.1f} GB "
          f"to download", flush=True)
    need = [(*f, sz[0], sz[2]) for f, sz in zip(files, sizes) if sz[1]]
    lock = threading.Lock(); done = [0]
    # one bar for the whole download: bytes against everything still missing, with speed, time left and files done
    with tqdm(total=todo, unit="B", unit_scale=True, unit_divisor=1024, desc="whole dataset", smoothing=0.05,
              postfix=f"0/{len(need)} files") as bar:
        def progress(n):
            with lock: bar.update(n)
        def finished(row):
            with lock: done[0] += 1; bar.set_postfix_str(f"{done[0]}/{len(need)} files")
        rows = s3_get_many(need, connections, progress, finished)
    return rows + [(f[2], sz[0], sz[2]) for f, sz in zip(files, sizes) if not sz[1]]


if __name__ == "__main__":                       # python -m retinopath.pipeline [--dev] [--connections N] [--list]
    import argparse
    from .core import load_config
    ap = argparse.ArgumentParser(description="Download all data needed by config.yaml")
    ap.add_argument("--dev", action="store_true"); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--list", action="store_true", help="only build the subject lists and count the scans")
    ap.add_argument("--connections", type=int, default=256, help="8 MB parts fetched at once, over all files")
    a = ap.parse_args()
    import resource                                    # each connection is an open file; Linux allows 1024 by default
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard if hard != resource.RLIM_INFINITY else 65536, hard))
    download_all(load_config(a.dev), a.workers, a.list, min(a.connections, 4096))
