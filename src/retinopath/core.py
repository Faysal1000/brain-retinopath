"""Config loading and cache-settings check, device, seeding, stage timer, atomic file writes."""

import csv, json, os, random, sys, time
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]

if sys.platform == "darwin":  # one torch thread: torch and dipy OpenMP crash together on macOS
    import torch

    torch.set_num_threads(1)


class Cfg(dict):
    """dict with attribute access."""

    __getattr__ = dict.get

    @staticmethod
    def wrap(value):
        if isinstance(value, dict):
            return Cfg({key: Cfg.wrap(item) for key, item in value.items()})
        return value


def load_config(dev_mode, path=None):
    """Read config.yaml and merge the `dev` or `full` block into the top level."""
    cfg = yaml.safe_load(open(path or ROOT / "config.yaml"))
    cfg.update(cfg["dev"] if dev_mode else cfg["full"])
    cfg["dev_mode"] = bool(dev_mode)
    cfg["protocols"] = {**cfg["protocols"], **cfg.get("extra_protocols", {})}
    cfg = Cfg.wrap(cfg)
    cfg["P"] = Cfg({key: (ROOT / value) for key, value in cfg.paths.items()})
    for key in ("data", "hcp", "osf", "nsd", "cache", "out"):
        cfg.P[key].mkdir(parents=True, exist_ok=True)
    if dev_mode:  # DEV runs use their own cache/dev and outputs/dev folders
        cfg.P["cache"] = cfg.P.cache / "dev"
        cfg.P["out"] = cfg.P.out / "dev"
        cfg.P.cache.mkdir(exist_ok=True)
        cfg.P.out.mkdir(exist_ok=True)
    check_cache_settings(cfg)
    return cfg


CACHED_KEYS = (
    "seed",
    "hemis",
    "protocols",
    "n_b0",
    "crop_margin_mm",
    "ref_voxel",
    "sh_order_out",
    "sh_order_in",
    "shells_in",
    "ref_model",
    "wm_mask_thr",
    "rumba_iters",
    "nsd.flip_x",
)


def check_cache_settings(cfg):
    """Record CACHED_KEYS in cache/settings.json; raise if the cache holds crops made with other values.
    A key missing from an older file does not count as a change."""
    setting = lambda key: cfg[key] if "." not in key else cfg[key.split(".")[0]][key.split(".")[1]]
    signature = json.dumps({key: setting(key) for key in CACHED_KEYS}, sort_keys=True, default=str)
    path = cfg.P.cache / "settings.json"
    if path.exists() and any(cfg.P.cache.glob("*/anat.npz")):
        old = path.read_text()
        if old != signature:
            old_values, new_values = json.loads(old), json.loads(signature)
            changed = [key for key in CACHED_KEYS if key in old_values and old_values[key] != new_values[key]]
            if changed:
                raise RuntimeError(
                    f"config.yaml changed in {changed} but {cfg.P.cache} holds files made with the old values. "
                    f"Delete that folder (or restore the old values) and run again."
                )
        else:
            return
    atomic(path, lambda temporary: Path(temporary).write_text(signature))


def get_device():
    """CUDA, else MPS, else CPU. oneDNN CPU kernels are off on Linux aarch64 (they fail on large 3D inputs)."""
    import platform, torch

    if platform.system() == "Linux" and platform.machine() == "aarch64":
        torch.backends.mkldnn.enabled = False
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def n_jobs(cfg):
    """CPU workers from cfg.n_jobs; negative values count back from all cores (-1 = all, -2 = all but one)."""
    requested = cfg.n_jobs
    return max(1, (os.cpu_count() or 2) + 1 + requested) if requested < 0 else requested


class Timer:
    """with Timer(cfg, "02_preprocessing"): ... appends the stage's hours to outputs/run_log.csv."""

    def __init__(self, cfg, stage):
        self.stage, self.log = stage, cfg.P.out / "run_log.csv"

    def total(self):
        return sum(float(row["hours"]) for row in csv.DictReader(open(self.log))) if self.log.exists() else 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        hours = (time.time() - self.t0) / 3600
        first_row = not self.log.exists()
        with open(self.log, "a", newline="") as handle:
            writer = csv.writer(handle)
            if first_row:
                writer.writerow(["stage", "hours", "finished", "ok"])
            writer.writerow([self.stage, f"{hours:.4f}", time.strftime("%Y-%m-%d %H:%M:%S"), exc[0] is None])
        print(f"[time] {self.stage}: {hours:.3f} h (all stages so far: {self.total():.2f} h)")
        return False


def atomic(path, write):
    """write(tmp) fills a temporary file that is then renamed onto `path`, so `path` is never left partial."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp{os.getpid()}_{path.name}")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def save_json(obj, path):
    atomic(path, lambda temporary: json.dump(obj, open(temporary, "w"), indent=1, default=float))


def load_json(path):
    return json.load(open(path))
