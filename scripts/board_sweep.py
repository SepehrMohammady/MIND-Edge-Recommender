"""Latency of the seven 8-bit encoders of Table 3 on a physical board, for two settings of ST's code
generator: its default (`-O balanced`) and `-O time` (schedule step 5).

For each encoder and setting: firmware/build.py builds ST's validation firmware and flashes it, then
scripts/board_validate.py --random runs `stedgeai validate --mode target` with the tool's 10 random inputs, as
ST's Developer Cloud does. A finished pair (its JSON in paper/results/board/) is not run again.

    python -m scripts.board_sweep --board STM32H7B3I-DK

Writes paper/results/board/<board>_sweep.json: per encoder the board's ms per title for both settings, the
activations size, and the Developer Cloud's ms for the same file and board (paper/results/stedgeai_cloud).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "paper/results"
MODELS = ["unas_h7", "unas_h7_c140", "unas_f401", "unas_f401_c142", "hand_64-2-384", "hand_32-5-384", "hand_64-5-384"]
FW = {"NUCLEO-F401RE": "nucleo_f401re", "STM32H7B3I-DK": "stm32h7b3i_dk"}
PY = str(ROOT / ".venv/Scripts/python.exe")


def result_path(board, model, opt):
    return RES / "board" / f"{board}__{model}_seed42_int8qdq{'_O' + opt if opt else ''}_random_validate.json"


def cloud_ms(board):
    out = {}
    for line in (RES / "stedgeai_cloud/runs.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("cmd") == "benchmark" and r.get("board") == board and (r.get("duration_ms") or -1) > 0:
            out[r["model"]] = r["duration_ms"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="STM32H7B3I-DK", choices=list(FW))
    a = ap.parse_args()
    env = {"PYTHONPATH": str(ROOT)}
    import os
    env = {**os.environ, **env}
    t0 = time.time()
    for opt in (None, "time"):
        for model in MODELS:
            if result_path(a.board, model, opt).exists():
                continue
            onnx = ROOT / f"artifacts/stedgeai/models/{model}_seed42_body_int8qdq.onnx"
            print(f"[{(time.time() - t0) / 60:5.1f}m] {model} {opt or 'balanced'}: build and flash", flush=True)
            b = subprocess.run([PY, str(ROOT / "firmware/build.py"), str(onnx), "--board", FW[a.board], "--app", "validation",
                                *(["--optimization", opt] if opt else []), "--flash"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
            if b.returncode != 0 or "Download verified successfully" not in b.stdout:
                print(b.stdout[-1500:], b.stderr[-1500:], flush=True)
                continue
            v = subprocess.run([PY, "-m", "scripts.board_validate", "--board", a.board, "--model", model, "--random",
                                *(["--optimization", opt] if opt else [])],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=ROOT)
            if v.returncode != 0:
                print(v.stdout[-1500:], v.stderr[-1500:], flush=True)
                continue
            r = json.loads(result_path(a.board, model, opt).read_text(encoding="utf-8"))
            print(f"[{(time.time() - t0) / 60:5.1f}m] {model} {opt or 'balanced'}: {r['duration_ms']} ms, "
                  f"activations {r['activations_bytes']} B, rmse {r['cross'].get('rmse')}", flush=True)

    cloud = cloud_ms(a.board)
    rows = []
    for model in MODELS:
        row = {"model": model, "cloud_ms": cloud.get(f"{model}_seed42_body_int8qdq.onnx")}
        for opt in (None, "time"):
            p = result_path(a.board, model, opt)
            if p.exists():
                r = json.loads(p.read_text(encoding="utf-8"))
                key = opt or "balanced"
                row[f"{key}_ms"] = r["duration_ms"]
                row[f"{key}_activations_bytes"] = r["activations_bytes"]
                row[f"{key}_outputs_equal"] = r["cross"].get("rmse") == "0.000000000"
                row[f"{key}_rmse"] = float(r["cross"].get("rmse"))
        rows.append(row)
    out = {"board": a.board, "inputs": "the tool's 10 random inputs", "generator": "ST Edge AI Core 4.0.0 (local), legacy C API",
           "cloud": "ST Edge AI Developer Cloud, Core 4.0.1, 2026-10-08/09", "rows": rows,
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RES / "board" / f"{a.board}_sweep.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for row in rows:
        print(row)


if __name__ == "__main__":
    sys.exit(main())
