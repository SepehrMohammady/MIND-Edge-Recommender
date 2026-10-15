"""On-target validation of an 8-bit encoder on a physical STM32 board, with real titles (schedule step 5).

The board runs ST's validation firmware (firmware/nucleo_f401re/build.py builds and flashes it). This script
embeds titles with the model's own byte table (on a device the caller does this lookup), writes them in the
validation format of ST Edge AI Core and runs `stedgeai validate --mode target`, which sends each title over
the ST-LINK serial port, runs the network on the board, reads back the output and the DWT cycle count, and
compares the outputs with the ONNX model run on the laptop. The board's outputs are also compared with ONNX
Runtime of this repository.

Titles: the first 12 English titles of MINDsmall dev, and the xMIND translations of the first 2 of them in
4 languages with four different scripts (Chinese, Japanese, Georgian, Finnish), 20 in all.

    python -m scripts.board_validate --model unas_f401_c142 --port COM3 --board NUCLEO-F401RE

Writes paper/results/board/<board>__<model>_validate.json and copies the tool's text report next to it.
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
import torch

from src import data_mind, data_xmind
from src.config import load_config
from src.student import text_to_bytes

ROOT = Path(__file__).resolve().parents[1]
STEDGEAI = Path(r"C:\ST\STEdgeAI\4.0\Utilities\windows\stedgeai.exe")
OUT = ROOT / "paper/results/board"
LANGS = ["zho", "jpn", "kat", "fin"]
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
        m = re.search(pattern, txt)
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
        "cross": dict(re.findall(r"(rmse|mae|l2r|cos|snr|nse)=([-\w.]+)", txt.split("Cross accuracy report #1")[-1])[:6]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unas_f401_c142")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--board", default="NUCLEO-F401RE")
    ap.add_argument("--target", default="stm32f4")
    a = ap.parse_args()

    onnx_path = ROOT / f"artifacts/stedgeai/models/{a.model}_seed{a.seed}_body_int8qdq.onnx"
    search = "mind_" + a.model[len("unas_"):] if a.model.startswith("unas_") else a.model
    ckpt = ROOT / f"artifacts/runs/unas_full/{search}_distill_ft_en_seed{a.seed}.pt"
    sd = torch.load(ckpt, map_location="cpu")
    table = sd["news_encoder.embed.weight"].numpy().astype(np.float32)

    rows = titles()
    L = cfg["data"]["max_title_bytes"]
    ids = np.array([text_to_bytes(t, L) for _, _, t in rows], dtype=np.int64)
    x = np.ascontiguousarray(table[ids].transpose(0, 2, 1)).astype(np.float32)      # (N, 64, 128)
    m = (ids != 0).astype(np.float32)[:, None, :]                                     # (N, 1, 128)

    work = ROOT / f"firmware/nucleo_f401re/build/{a.model[len('unas_'):] if a.model.startswith('unas_') else a.model}"
    vdir = work / "validate_titles"
    vdir.mkdir(parents=True, exist_ok=True)
    npz = vdir / "titles_inputs.npz"
    np.savez(npz, m_inputs_1=x.reshape(len(x), 64, 1, 128), m_inputs_2=m.reshape(len(m), 128, 1, 1))

    t0 = time.time()
    cmd = [STEDGEAI, "validate", "-m", onnx_path, "--target", a.target, "--c-api", "legacy", "--name", "network",
           "--mode", "target", "-d", f"serial:{a.port}:115200", "--valinput", npz,
           "--output", vdir, "--workspace", work / "ws_val"]
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit("stedgeai validate failed")
    report = (vdir / "network_validate_report.txt").read_text(encoding="utf-8", errors="replace")
    res = parse_report(report)

    io = np.load(vdir / "network_val_io.npz")
    c_out = io["c_outputs_1"].reshape(len(rows), 384).astype(np.float32)
    s = re.search(r"QLinear\(([\d.e-]+),(-?\d+),int8\)", report)
    scale, zp = float(s.group(1)), int(s.group(2))
    board = (c_out - zp) * scale
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    ref = np.stack([sess.run(None, {names[0]: x[i:i + 1], names[1]: m[i:i + 1]})[0][0] for i in range(len(rows))])
    cos = (board * ref).sum(1) / (np.linalg.norm(board, axis=1) * np.linalg.norm(ref, axis=1))

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{a.board}__{a.model}_seed{a.seed}_int8qdq"
    shutil.copy(vdir / "network_validate_report.txt", OUT / f"{stem}_validate_report.txt")
    result = {
        "board": a.board, "model": onnx_path.name, "checkpoint": str(ckpt.relative_to(ROOT)),
        "firmware": "ST validation application (ST Edge AI Core 4.0.0, legacy C API), firmware/nucleo_f401re",
        "titles": [{"lang": lg, "nid": n} for lg, n, _ in rows],      # news ids only (the text stays in MIND/xMIND)
        **res,
        "board_vs_onnxruntime": {"onnxruntime": ort.__version__, "max_abs_diff": float(np.abs(board - ref).max()),
                                 "min_cosine": float(cos.min()), "titles": len(rows)},
        "minutes": round((time.time() - t0) / 60, 2), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUT / f"{stem}_validate.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("device", "duration_ms", "cycles", "cycles_per_macc", "cross",
                                             "board_vs_onnxruntime")}, indent=1))


if __name__ == "__main__":
    main()
