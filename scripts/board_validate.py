"""On-target validation of an 8-bit encoder on a physical STM32 board (schedule step 5).

The board runs ST's validation firmware (firmware/build.py builds and flashes it). The script runs
`stedgeai validate --mode target`, which sends each input over the ST-LINK serial port, runs the network on
the board, reads back the output and the DWT cycle count, and compares the outputs with the ONNX model run on
the laptop; the board's outputs are also compared with ONNX Runtime of this repository.

Inputs: real titles, embedded with the model's own byte table (on a device the caller does this lookup): the
first 12 English titles of MINDsmall dev and the xMIND translations of the first 2 of them in 4 languages
with four different scripts (Chinese, Japanese, Georgian, Finnish), 20 in all; or, with --random, the
tool's 10 random inputs (what ST's Developer Cloud uses).

    python -m scripts.board_validate --board NUCLEO-F401RE --model unas_f401_c142
    python -m scripts.board_validate --board STM32H7B3I-DK --model unas_h7_c140 --optimization time [--random]

The --optimization value must be the one the flashed firmware was generated with (firmware/build.py).
Writes paper/results/board/<board>__<model>_seed<s>_int8qdq[_O<opt>][_random]_validate.json and copies the
tool's text report next to it.
"""
import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src import board_models, data_mind, data_xmind, runlog
from src.config import load_config
from src.student import text_to_bytes

ROOT = Path(__file__).resolve().parents[1]
STEDGEAI = Path(r"C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe")
OUT = ROOT / "paper/results/board"
LANGS = ["zho", "jpn", "kat", "fin"]
BOARDS = {"NUCLEO-F401RE": ("nucleo_f401re", "stm32f4", "COM3"), "STM32H7B3I-DK": ("stm32h7b3i_dk", "stm32h7", "COM6")}
cfg = load_config()


def titles():
    dev = data_mind.read_news(cfg, "dev")
    nids = [n for n, v in dev.items() if v["title"].strip()][:12]
    out = [("eng", n, dev[n]["title"]) for n in nids]
    for lang in LANGS:
        tr = data_xmind.read_xmind_text(cfg, lang, "dev")
        out += [(lang, n, tr[n]["title"]) for n in nids[:2] if n in tr]
    return out


def parse_report(txt):
    def grab(pattern, cast=float):
        m = re.search(pattern, txt, re.M)
        return cast(m.group(1).replace(",", "")) if m else None
    return {
        "device": (re.search(r"device\.desc\s*:\s*(.+?)\s*$", txt, re.M) or [None, None])[1],
        "device_attrs": (re.search(r"device\.attrs\s*:\s*(.+?)\s*$", txt, re.M) or [None, None])[1],
        "runtime": (re.search(r"runtime lib\s*:\s*(.+?)\s*$", txt, re.M) or [None, None])[1],
        "samples": grab(r"nb sample\(s\)\s*:\s*(\d+)", int),
        "duration_ms": grab(r"duration\s*:\s*([\d.]+) ms by sample"),
        "duration_min_max_std": (re.search(r"ms by sample \(([\d./]+)\)", txt) or [None, None])[1],
        "cycles": grab(r"CPU cycles\s*:\s*\[([\d,]+)\]", int),
        "cycles_per_macc": grab(r"cycles/MACC\s*:\s*([\d.]+)"),
        "macc": grab(r"^\s*macc\s*:\s*([\d,]+)", int),
        "activations_bytes": grab(r"^\s*activations\s*:\s*(\d+)", int),
        "weights_bytes": grab(r"^\s*weights\s*:\s*(\d+)", int),
        "cross": dict(re.findall(r"(rmse|mae|l2r|cos|snr|nse)=([-\w.]+)", txt.split("Cross accuracy report #1")[-1])[:6]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unas_f401_c142")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--board", default="NUCLEO-F401RE", choices=list(BOARDS))
    ap.add_argument("--port", default=None)
    ap.add_argument("--optimization", default=None, help="stedgeai -O of the flashed firmware (default balanced)")
    ap.add_argument("--random", action="store_true", help="the tool's random inputs instead of titles")
    ap.add_argument("--reparse", action="store_true", help="rebuild the JSON from the last run's files, no board")
    a = ap.parse_args()
    fw, target, port = BOARDS[a.board]
    port = a.port or port

    onnx_path = ROOT / f"artifacts/stedgeai/models/{a.model}_seed{a.seed}_body_int8qdq.onnx"
    ckpt = board_models.checkpoint_for(a.model, a.seed)
    opt = f"_O{a.optimization}" if a.optimization else ""
    work = ROOT / f"firmware/{fw}/build/{a.model}{opt}"
    vdir = work / ("validate_random" if a.random else "validate_titles")
    vdir.mkdir(parents=True, exist_ok=True)

    rows = []
    if not a.random:
        table = board_models.byte_table(a.model, a.seed)
        rows = titles()
        L = cfg["data"]["max_title_bytes"]
        ids = np.array([text_to_bytes(t, L) for _, _, t in rows], dtype=np.int64)
        x = np.ascontiguousarray(table[ids].transpose(0, 2, 1)).astype(np.float32)      # (N, 64, 128)
        m = (ids != 0).astype(np.float32)[:, None, :]                                     # (N, 1, 128)
        npz = vdir / "titles_inputs.npz"
        np.savez(npz, m_inputs_1=x.reshape(len(x), 64, 1, 128), m_inputs_2=m.reshape(len(m), 128, 1, 1))

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{a.board}__{a.model}_seed{a.seed}_int8qdq{opt}{'_random' if a.random else ''}"
    old = OUT / f"{stem}_validate.json"
    cmd = [STEDGEAI, "validate", "-m", onnx_path, "--target", target, "--c-api", "legacy", "--name", "network",
           *(["-O", a.optimization] if a.optimization else []), "--mode", "target", "-d", f"serial:{port}:115200",
           *([] if a.random else ["--valinput", npz]), "--output", vdir, "--workspace", work / "ws_val"]
    if not a.reparse:
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
            raise SystemExit("stedgeai validate failed")
    report = (vdir / "network_validate_report.txt").read_text(encoding="utf-8", errors="replace")
    res = parse_report(report)

    io = np.load(vdir / "network_val_io.npz")
    if a.random:
        x = io["m_inputs_1"].reshape(-1, 64, 128).astype(np.float32)
        m = io["m_inputs_2"].reshape(-1, 1, 128).astype(np.float32)
    n = len(x)
    c_out = io["c_outputs_1"].reshape(n, 384).astype(np.float32)
    s = re.findall(r"QLinear\(([\d.e-]+),(-?\d+),int8\)", report.split("O[1/1]")[-1])[0]
    scale, zp = float(s[0]), int(s[1])
    board = (c_out - zp) * scale
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    ref = np.stack([sess.run(None, {names[0]: x[i:i + 1], names[1]: m[i:i + 1]})[0][0] for i in range(n)])
    cos = (board * ref).sum(1) / (np.linalg.norm(board, axis=1) * np.linalg.norm(ref, axis=1) + 1e-12)

    shutil.copy(vdir / "network_validate_report.txt", OUT / f"{stem}_validate_report.txt")
    result = {
        "board": a.board, "model": onnx_path.name, "checkpoint": str(ckpt.relative_to(ROOT)),
        "firmware": f"ST validation application (ST Edge AI Core 4.0.0, legacy C API), firmware/{fw}",
        "optimization": a.optimization or "balanced", "inputs": "random (the tool's)" if a.random else "titles",
        "titles": [{"lang": lg, "nid": nid} for lg, nid, _ in rows],      # news ids only (the text stays in MIND/xMIND)
        **res,
        "board_vs_onnxruntime": {"onnxruntime": ort.__version__, "max_abs_diff": float(np.abs(board - ref).max()),
                                 "min_cosine": float(cos.min()), "samples": n},
        "minutes": round((time.time() - t0) / 60, 2), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        **({k: json.loads(old.read_text(encoding="utf-8"))[k] for k in ("minutes", "finished")} if a.reparse else {}),
    }
    (OUT / f"{stem}_validate.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    if not a.reparse:
        cfg["seed"] = a.seed
        runlog.append(cfg, f"board/{stem}_validate", {"port": port, "firmware": result["firmware"],
                      "optimization": result["optimization"], "inputs": result["inputs"]},
                      {k: result[k] for k in ("device", "duration_ms", "cycles", "cycles_per_macc", "activations_bytes",
                                              "cross", "board_vs_onnxruntime")}, t0)
    print(json.dumps({k: result[k] for k in ("device", "optimization", "inputs", "duration_ms", "cycles",
                                             "activations_bytes", "cross", "board_vs_onnxruntime")}, indent=1))


if __name__ == "__main__":
    main()
