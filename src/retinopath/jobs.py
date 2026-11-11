"""Run independent jobs on everything available: all local GPUs, or all nodes listed in config.yaml."""
import os, signal, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from .core import ROOT


def n_gpus():
    import torch
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def train_commands(cfg):
    """(name, python-args) per training run: `path` and `voxel` run once per entry of train_seeds, other variants
    only with the first seed."""
    runs = [(v, s) for v in cfg.variants for s in (cfg.train_seeds if v in ("path", "voxel") else cfg.train_seeds[:1])]
    dev = ["--dev"] if cfg.dev_mode else []
    return [(f"{v}_s{s}", ["-m", "retinopath.train", "--variant", v, "--seed", str(s)] + dev) for v, s in runs]


def run_jobs(cfg, jobs):
    """jobs: list of (name, python-args). Slots = nodes x GPUs. Each slot runs its jobs one after another.
    With `multi_gpu: ddp` and one machine, every job instead uses all local GPUs through torchrun."""
    logs = cfg.P.out / "logs"; logs.mkdir(exist_ok=True)
    g, nodes = n_gpus(), list(cfg.nodes or [])
    py = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv").exists() else sys.executable
    if not nodes and g > 1 and cfg.multi_gpu == "ddp":
        slots, prefix = [None], [py, "-m", "torch.distributed.run", f"--nproc_per_node={g}"]
    else:
        slots = [(n, i) for n in (nodes or [None]) for i in range(max(g, 1))]; prefix = [py]

    procs, stop = [], {"now": False}

    def worker(k):
        res = []
        for name, args in jobs[k::len(slots)]:
            if stop["now"]:
                break
            cmd, env = prefix + args, dict(os.environ)
            if slots[k] is not None:
                node, gpu = slots[k]
                if g > 1:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                if node:      # same project path on every node, passwordless SSH
                    cmd = ["ssh", node, f"cd {ROOT} && CUDA_VISIBLE_DEVICES={gpu} " + " ".join(cmd)]
            with open(logs / f"{name}.log", "a") as f:
                p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                procs.append(p); rc = p.wait()
            res.append((name, rc))                       # rc 130: train.py saved a checkpoint after SIGINT/SIGTERM
            print(f"{name}: {'ok' if rc == 0 else 'stopped, checkpoint saved' if rc == 130 else 'FAILED, see ' + str(logs / (name + '.log'))}", flush=True)
        return res

    with ThreadPoolExecutor(len(slots)) as ex:
        futs = [ex.submit(worker, k) for k in range(len(slots))]
        try:
            return [r for f in futs for r in f.result()]
        except KeyboardInterrupt:          # notebook interrupted: ask every running training to save, then stop
            stop["now"] = True
            for p in procs:
                if p.poll() is None:
                    p.send_signal(signal.SIGINT)
            for p in procs:
                try:
                    p.wait(120)
                except subprocess.TimeoutExpired:
                    p.kill()
            print("interrupted: local training runs saved a checkpoint; run this notebook again to continue", flush=True)
            raise
