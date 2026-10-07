"""ST Edge AI Developer Cloud: analyze and benchmark the exported encoders on real
boards, through the Python client vendored in ST's model-zoo services (the same
procedure the lab uses for the Lane-Change-MCU models).

Credentials are read from the environment (STEDGEAI_USER, STEDGEAI_PASS) and are
never written to disk by this script; the token file the client caches in the
home directory is deleted on exit.

    python -m scripts.stedgeai_cloud boards
    python -m scripts.stedgeai_cloud analyze  <model.onnx>
    python -m scripts.stedgeai_cloud benchmark <model.onnx> <board name> [<board name> ...]

Results (no credentials inside) go to paper/results/stedgeai_cloud/<stem>__<what>.json
and every call appends a line to paper/results/stedgeai_cloud/runs.jsonl.
"""
import json
import os
import pathlib
import sys
import time

CLIENT = r"C:/Projects/PhD/DIMIR/Materials/stm32ai-modelzoo-services"
sys.path.insert(0, CLIENT)
from common.stm32ai_dc import CliParameters, CloudBackend, Stm32Ai  # noqa: E402

OUT = pathlib.Path("paper/results/stedgeai_cloud")
OUT.mkdir(parents=True, exist_ok=True)


def jsonable(x):
    if hasattr(x, "_asdict"):
        return {k: jsonable(v) for k, v in x._asdict().items()}
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def connect():
    # With both variables unset the client falls back to the token cached in the home directory.
    user, pw = os.environ.get("STEDGEAI_USER"), os.environ.get("STEDGEAI_PASS")
    backend = CloudBackend(user, pw, version=os.environ.get("STEDGEAI_VERSION"))
    return Stm32Ai(backend), backend.version


TOKEN = pathlib.Path.home() / ".stmai_token"
TOKEN_PREEXISTED = TOKEN.exists()


def cleanup():
    # a token cached by an earlier session on this machine is left alone; one created here is removed
    if TOKEN.exists() and not TOKEN_PREEXISTED:
        TOKEN.unlink()


def log(record):
    with open(OUT / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def summary_analyze(r):
    return {"macc": r.macc, "weights_bytes": r.weights, "activations_bytes": r.activations_size,
            "rom_bytes": r.rom_size, "ram_bytes": r.ram_size, "io_bytes": r.total_ram_io_size,
            "library_flash_bytes": r.estimated_library_flash_size, "library_ram_bytes": r.estimated_library_ram_size,
            "tool": r.cli_version_str, "date": r.date_time}


def summary_benchmark(r):
    s = summary_analyze(r)
    s.update({"board": r.device, "duration_ms": r.duration_ms, "cycles": r.cycles, "cycles_by_macc": r.cycles_by_macc,
              "internal_ram_bytes": r.internal_ram_consumption, "external_ram_bytes": r.external_ram_consumption,
              "uses_external_ram": r.use_external_ram, "uses_external_flash": r.use_external_flash,
              "validation_error": r.validation_error, "validation_error_description": r.validation_error_description})
    return s


def main():
    cmd = sys.argv[1]
    ai, version = connect()
    try:
        if cmd == "boards":
            for b in ai.get_benchmark_boards():
                print(jsonable(b))
            print("tool version:", version)
        elif cmd in ("analyze", "benchmark"):
            path = pathlib.Path(sys.argv[2])
            name = path.name
            t0 = time.time()
            ai.upload_model(str(path))
            print("uploaded", name)
            if cmd == "analyze":
                res = ai.analyze(CliParameters(model=name))
                s = summary_analyze(res)
                (OUT / f"{path.stem}__analyze.json").write_text(json.dumps(jsonable(res), indent=1), encoding="utf-8")
                log({"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "cmd": cmd, "model": name, "tool": version,
                     "minutes": round((time.time() - t0) / 60, 2), **s})
                print(json.dumps(s, indent=1))
            else:
                for board in sys.argv[3:]:
                    t1 = time.time()
                    res = ai.benchmark(CliParameters(model=name), board_name=board, timeout=2500)
                    s = summary_benchmark(res)
                    (OUT / f"{path.stem}__{board}.json").write_text(json.dumps(jsonable(res), indent=1), encoding="utf-8")
                    log({"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "cmd": cmd, "model": name, "tool": version,
                         "minutes": round((time.time() - t1) / 60, 2), **s})
                    print(board, json.dumps(s, indent=1))
        else:
            sys.exit("unknown command")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
