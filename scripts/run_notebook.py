"""Execute a notebook without a browser and print its text output while it runs (progress bars, training lines,
errors). The executed copy, with all outputs, is written to `out`.

    python scripts/run_notebook.py notebooks/05_training.ipynb outputs/executed/05_training.ipynb [--dev True|False]

--dev sets the DEV_MODE line of the first code cell in the copy; without it the notebook's own value is used.
Progress bars are drawn as text: the widget bars of tqdm.auto need a browser and would print nothing here.
"""
import argparse, re, sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

TEXT_BARS = "import tqdm.auto, tqdm.std; tqdm.auto.tqdm, tqdm.auto.trange = tqdm.std.tqdm, tqdm.std.trange"
# joblib keeps idle worker processes; stopped by the kernel shutdown they each print a KeyboardInterrupt traceback
STOP_WORKERS = "from joblib.externals.loky import get_reusable_executor; get_reusable_executor().shutdown(wait=True)"


class LiveClient(NotebookClient):
    def on_cell_start_(self, cell, cell_index):
        n = sum(c.cell_type == "code" for c in self.nb.cells) - 2            # the two inserted cells are not counted
        k = sum(c.cell_type == "code" for c in self.nb.cells[:cell_index])   # code cells before this one
        if 0 < k <= n and cell.cell_type == "code":
            first = cell.source.strip().splitlines()[0][:90] if cell.source.strip() else ""
            print(f"--- cell {k}/{n}: {first}", flush=True)

    def output(self, outs, msg, display_id, cell_index):
        c, kind = msg["content"], msg["msg_type"]
        if kind == "stream":
            (sys.stderr if c["name"] == "stderr" else sys.stdout).write(c["text"])
            (sys.stderr if c["name"] == "stderr" else sys.stdout).flush()
        elif kind == "error":
            print("\n".join(c["traceback"]), file=sys.stderr, flush=True)
        return super().output(outs, msg, display_id, cell_index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook"); ap.add_argument("out"); ap.add_argument("--dev", choices=["True", "False"])
    a = ap.parse_args()
    src = Path(a.notebook).resolve(); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(src, 4)
    if a.dev:
        c = next(c for c in nb.cells if c.cell_type == "code")
        c.source = re.sub(r"^DEV_MODE = (True|False)", f"DEV_MODE = {a.dev}", c.source, count=1, flags=re.M)
        assert f"DEV_MODE = {a.dev}" in c.source, "first code cell has no DEV_MODE line"
    nb.cells.insert(0, nbformat.v4.new_code_cell(TEXT_BARS)); nb.cells.append(nbformat.v4.new_code_cell(STOP_WORKERS))
    client = LiveClient(nb, kernel_name="retinopath", timeout=None, resources={"metadata": {"path": str(src.parent)}})
    client.on_cell_start = client.on_cell_start_
    try:
        client.execute()
        ok = True
    except CellExecutionError:
        ok = False
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr); ok = False
    finally:
        nb.cells.pop(0); nb.cells.pop()
        nbformat.write(nb, out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
