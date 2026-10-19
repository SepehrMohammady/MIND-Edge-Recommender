"""Results of the one-bit firmware (firmware/common/Src/main_bnn.cc) from its serial log (schedule step 5).

    python -m scripts.board_bnn logs/board_bnn_h7.log [--board STM32H7B3I-DK] [--model reactnet_64-5-384_seed42]

The log holds, per test title, the DWT cycles of the byte-table lookup and of Invoke(), and the 384 outputs as
hexadecimal floats; then the timing over 10 more runs of each title and the mean cycles of each operator. The
script compares the board's outputs with LCE's interpreter on the laptop and with PyTorch (artifacts/binary_lce),
and writes paper/results/board/<board>__<model>_lce.json; the log is copied next to it.
"""
import argparse
import json
import re
import shutil
import struct
import time
from pathlib import Path

import numpy as np

from src import runlog
from src.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--board", default="STM32H7B3I-DK")
    ap.add_argument("--model", default="reactnet_64-5-384_seed42")
    a = ap.parse_args()
    log = Path(a.log).read_text(encoding="utf-8-sig").splitlines()
    body = [l[25:] if re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}  ", l) else l for l in log]
    head = next(l for l in body if l.startswith("bnn: "))
    hz = int(re.search(r"CPU (\d+) Hz", head).group(1))
    arena = re.search(r"arena used (\d+) of (\d+) B", head)
    model_b = int(re.search(r"model (\d+) B", head).group(1))
    titles = {}
    for l in body:
        m = re.match(r"title (\d+) lookup (\d+) cycles invoke (\d+) cycles maxdiff_e9 (\d+)", l)
        if m:
            titles[int(m.group(1))] = {"lookup_cycles": int(m.group(2)), "invoke_cycles": int(m.group(3)),
                                       "board_maxdiff_vs_lce": int(m.group(4)) * 1e-9}
    n = len(titles)
    out = np.zeros((n, 384), dtype=np.float32)
    for l in body:
        m = re.match(r"out (\d+) (\d+) ((?:[0-9a-f]{8} ?){16})", l)
        if m:
            t, i = int(m.group(1)), int(m.group(2))
            out[t, i:i + 16] = [struct.unpack("<f", struct.pack("<I", int(h, 16)))[0] for h in m.group(3).split()]
    d = ROOT / "artifacts/binary_lce"
    lce = np.load(d / f"{a.model}_lce_test_out.npy")[:n]
    torch_out = np.load(d / f"{a.model}_test.npz")["torch_out"][:n]

    def cmp(x, y):
        cos = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)
        return {"max_abs_diff": float(np.abs(x - y).max()), "min_cosine": float(cos.min())}
    bench = re.search(r"bench: (\d+) runs, invoke mean (\d+) min (\d+) max (\d+) cycles",
                      next(l for l in body if l.startswith("bench: ")))
    ops = []
    for l in body:
        m = re.match(r"op (\d+) (\S+) mean (\d+) cycles", l)
        if m:
            ops.append({"index": int(m.group(1)), "op": m.group(2), "cycles": int(m.group(3))})
    by_type = {}
    for o in ops:
        by_type[o["op"]] = by_type.get(o["op"], 0) + o["cycles"]
    mean_cycles = int(bench.group(2))
    res = {
        "board": a.board, "model": f"{a.model}_lce.tflite", "runtime": "TensorFlow Lite Micro + Larq Compute Engine "
        "reference kernels (NAS-BNN's port, zero-padding fix of firmware/fetch_deps.py)", "cpu_hz": hz,
        "model_bytes": model_b, "arena_used_bytes": int(arena.group(1)), "titles": n,
        "invoke": {"runs": int(bench.group(1)), "mean_cycles": mean_cycles, "min_cycles": int(bench.group(3)),
                   "max_cycles": int(bench.group(4)), "mean_ms": round(1000 * mean_cycles / hz, 3)},
        "lookup_ms_mean": round(1000 * np.mean([t["lookup_cycles"] for t in titles.values()]) / hz, 3),
        "board_vs_lce_interpreter": cmp(out, lce), "board_vs_pytorch": cmp(out, torch_out),
        "ops": ops, "cycles_by_op_type": by_type,
        "share_by_op_type": {k: round(v / mean_cycles, 3) for k, v in by_type.items()},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    dst = ROOT / "paper/results/board" / f"{a.board}__{a.model}_lce.json"
    dst.write_text(json.dumps(res, indent=1), encoding="utf-8")
    shutil.copy(a.log, dst.with_name(f"{a.board}__{a.model}_lce_serial.log"))
    cfg = load_config()
    cfg["seed"] = 42
    runlog.append(cfg, f"board/{dst.stem}", {"runtime": res["runtime"], "log": Path(a.log).name},
                  {k: res[k] for k in ("invoke", "arena_used_bytes", "board_vs_lce_interpreter", "board_vs_pytorch",
                                       "share_by_op_type")}, time.time())
    print(json.dumps({k: res[k] for k in ("invoke", "lookup_ms_mean", "arena_used_bytes", "board_vs_lce_interpreter",
                                          "board_vs_pytorch", "share_by_op_type")}, indent=1))


if __name__ == "__main__":
    main()
