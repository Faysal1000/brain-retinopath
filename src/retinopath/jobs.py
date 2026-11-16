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
    runs = [
        (variant, seed)
        for variant in cfg.variants
        for seed in (cfg.train_seeds if variant in ("path", "voxel") else cfg.train_seeds[:1])
    ]
    dev = ["--dev"] if cfg.dev_mode else []
    return [
        (f"{variant}_s{seed}", ["-m", "retinopath.train", "--variant", variant, "--seed", str(seed)] + dev)
        for variant, seed in runs
    ]


def run_jobs(cfg, jobs):
    """jobs: list of (name, python-args). Slots = nodes x GPUs. Each slot runs its jobs one after another.
    With `multi_gpu: ddp` and one machine, every job instead uses all local GPUs through torchrun."""
    logs = cfg.P.out / "logs"
    logs.mkdir(exist_ok=True)
    gpus, nodes = n_gpus(), list(cfg.nodes or [])
    python = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv").exists() else sys.executable
    if not nodes and gpus > 1 and cfg.multi_gpu == "ddp":
        slots, prefix = [None], [python, "-m", "torch.distributed.run", f"--nproc_per_node={gpus}"]
    else:
        slots = [(node, gpu_index) for node in (nodes or [None]) for gpu_index in range(max(gpus, 1))]
        prefix = [python]

    procs, stop = [], {"now": False}

    def worker(slot):
        results = []
        for name, args in jobs[slot :: len(slots)]:
            if stop["now"]:
                break
            cmd, env = prefix + args, dict(os.environ)
            if slots[slot] is not None:
                node, gpu = slots[slot]
                if gpus > 1:
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                if node:  # same project path on every node, passwordless SSH
                    cmd = ["ssh", node, f"cd {ROOT} && CUDA_VISIBLE_DEVICES={gpu} " + " ".join(cmd)]
            print(
                f"{name}: started (run {jobs.index((name, args)) + 1} of {len(jobs)}), log in {logs / (name + '.log')}",
                flush=True,
            )
            # every line goes to the log file and, prefixed with the run name, to the notebook (or terminal)
            with open(logs / f"{name}.log", "a") as log_file:
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                    bufsize=1,
                )
                procs.append(process)
                for line in process.stdout:
                    log_file.write(line)
                    log_file.flush()
                    print(f"[{name}] {line}", end="", flush=True)
                return_code = process.wait()
            results.append((name, return_code))  # rc 130: train.py saved a checkpoint after SIGINT/SIGTERM
            print(
                f"{name}: {'ok' if return_code == 0 else 'stopped, checkpoint saved' if return_code == 130 else 'FAILED, see ' + str(logs / (name + '.log'))}",
                flush=True,
            )
        return results

    with ThreadPoolExecutor(len(slots)) as executor:
        futures = [executor.submit(worker, slot) for slot in range(len(slots))]
        try:
            return [result for future in futures for result in future.result()]
        except KeyboardInterrupt:  # notebook interrupted: ask every running training to save, then stop
            stop["now"] = True
            for process in procs:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
            for process in procs:
                try:
                    process.wait(120)
                except subprocess.TimeoutExpired:
                    process.kill()
            print(
                "interrupted: local training runs saved a checkpoint; run this notebook again to continue", flush=True
            )
            raise
