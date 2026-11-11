"""Config loading and cache-settings check, device, seeding, stage timer, atomic file writes."""
import csv, json, os, random, sys, time
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

if sys.platform == "darwin":      # macOS: torch CPU threads must be limited after dipy's OpenMP code has run, otherwise it segfaults
    import torch
    torch.set_num_threads(1)


class Cfg(dict):
    """dict with attribute access."""
    __getattr__ = dict.get

    @staticmethod
    def wrap(x):
        if isinstance(x, dict):
            return Cfg({k: Cfg.wrap(v) for k, v in x.items()})
        return x


def load_config(dev_mode, path=None):
    """Read config.yaml and merge the `dev` or `full` block into the top level."""
    cfg = yaml.safe_load(open(path or ROOT / "config.yaml"))
    cfg.update(cfg["dev"] if dev_mode else cfg["full"])
    cfg["dev_mode"] = bool(dev_mode)
    cfg["protocols"] = {**cfg["protocols"], **cfg.get("extra_protocols", {})}
    cfg = Cfg.wrap(cfg)
    cfg["P"] = Cfg({k: (ROOT / v) for k, v in cfg.paths.items()})
    for k in ("data", "hcp", "osf", "nsd", "cache", "out"):
        cfg.P[k].mkdir(parents=True, exist_ok=True)
    if dev_mode:  # DEV runs use their own cache/dev and outputs/dev folders
        cfg.P["cache"] = cfg.P.cache / "dev"; cfg.P["out"] = cfg.P.out / "dev"
        cfg.P.cache.mkdir(exist_ok=True); cfg.P.out.mkdir(exist_ok=True)
    check_cache_settings(cfg)
    return cfg


CACHED_KEYS = ("seed", "hemis", "protocols", "n_b0", "crop_margin_mm", "ref_voxel", "sh_order_out", "sh_order_in", "shells_in",
               "ref_model", "wm_mask_thr", "rumba_iters", "tracking", "seeds_per_vertex", "nsd.seeds_per_vertex", "nsd.flip_x")


def check_cache_settings(cfg):
    """Cached crops, fODFs and tractograms depend on CACHED_KEYS. Their values are recorded in cache/settings.json;
    if the cache already holds a crop (*/anat.npz) and any of them differs, raise instead of mixing old and new
    results. Otherwise the current values are (re)written. A key the old file does not know is not a change."""
    get = lambda k: cfg[k] if "." not in k else cfg[k.split(".")[0]][k.split(".")[1]]
    sig = json.dumps({k: get(k) for k in CACHED_KEYS}, sort_keys=True, default=str)
    f = cfg.P.cache / "settings.json"
    if f.exists() and any(cfg.P.cache.glob("*/anat.npz")):
        old = f.read_text()
        if old != sig:
            o, n = json.loads(old), json.loads(sig)
            diff = [k for k in CACHED_KEYS if k in o and o[k] != n[k]]
            if diff:
                raise RuntimeError(f"config.yaml changed in {diff} but {cfg.P.cache} holds files made with the old values. "
                                   f"Delete that folder (or restore the old values) and run again.")
        else:
            return
    atomic(f, lambda t: Path(t).write_text(sig))            # several processes may start at the same moment


def get_device():
    """CUDA if available, else MPS, else CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def n_jobs(cfg):
    """CPU workers from cfg.n_jobs; negative values count back from all cores (-1 = all, -2 = all but one)."""
    n = cfg.n_jobs
    return max(1, (os.cpu_count() or 2) + 1 + n) if n < 0 else n


class Timer:
    """Context manager that appends a stage's wall-clock hours to outputs/run_log.csv (stage, hours, finished, ok).

    with Timer(cfg, "02_preprocessing"): ...
    """

    def __init__(self, cfg, stage):
        self.stage, self.log = stage, cfg.P.out / "run_log.csv"

    def total(self):
        return sum(float(r["hours"]) for r in csv.DictReader(open(self.log))) if self.log.exists() else 0.0

    def __enter__(self):
        self.t0 = time.time(); return self

    def __exit__(self, *exc):
        h = (time.time() - self.t0) / 3600
        new = not self.log.exists()
        with open(self.log, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["stage", "hours", "finished", "ok"])
            w.writerow([self.stage, f"{h:.4f}", time.strftime("%Y-%m-%d %H:%M:%S"), exc[0] is None])
        print(f"[time] {self.stage}: {h:.3f} h (all stages so far: {self.total():.2f} h)")
        return False


def atomic(path, write):
    """Atomic write: `write(tmp)` fills a temporary file in the same folder, which os.replace moves onto `path` only
    after `write` returns. An interrupted run never leaves a partial file at `path`; the temporary file is removed."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp{os.getpid()}_{path.name}")
    try:
        write(tmp); os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def save_json(obj, path):
    atomic(path, lambda t: json.dump(obj, open(t, "w"), indent=1, default=float))


def load_json(path):
    return json.load(open(path))
