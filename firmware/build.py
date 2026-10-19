"""Build the board firmware for one 8-bit encoder, and optionally flash it.

    python firmware/build.py --board nucleo_f401re|stm32h7b3i_dk MODEL.onnx [--app validation|energy] [--flash]

  validation  ST's validation application; `stedgeai validate --mode target` then talks to the board
              (scripts/board_validate.py)
  energy      the encoder in a loop on one embedded title, and STOP mode, switched with the user button
              (common/Src/main_energy.c); the title's input and the laptop's 8-bit output are written to
              <board>/build/<tag>/energy/title_input.h

Steps: `stedgeai generate` (local ST Edge AI Core 4.0, legacy C API, default options) writes the network
code to <board>/build/<tag>/gen; arm-none-eabi-gcc (GNU Tools for STM32 14.3 of STM32CubeIDE 2.1.1)
compiles it with the application (common/), the board files (<board>/Core), the STM32 HAL and CMSIS
(third_party, fetch_deps.py) and links ST's network runtime library; STM32_Programmer_CLI flashes the ELF
over the board's ST-LINK.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TP = HERE / "third_party"
STEDGEAI_ROOT = Path(r"C:\ST\STEdgeAI\4.0")
STEDGEAI = STEDGEAI_ROOT / "Utilities/windows/stedgeai.exe"
AI = STEDGEAI_ROOT / "Middlewares/ST/AI"
GCC_BIN = Path(r"C:\ST\STM32CubeIDE_2.1.1\STM32CubeIDE\plugins"
               r"\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740"
               r"\tools\bin")
PROGRAMMER = Path(r"C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe")
CORE = TP / "CMSIS_5/CMSIS/Core/Include"

BOARDS = {
    "nucleo_f401re": {
        "target": "stm32f4", "defines": ["-DSTM32F401xE"],
        "cpu": ["-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"],
        "hal": TP / "stm32f4xx-hal-driver", "dev": TP / "cmsis-device-f4",
        "hal_src": ["stm32f4xx_hal.c", "stm32f4xx_hal_cortex.c", "stm32f4xx_hal_rcc.c", "stm32f4xx_hal_rcc_ex.c",
                    "stm32f4xx_hal_gpio.c", "stm32f4xx_hal_uart.c", "stm32f4xx_hal_dma.c", "stm32f4xx_hal_dma_ex.c",
                    "stm32f4xx_hal_crc.c", "stm32f4xx_hal_pwr.c", "stm32f4xx_hal_pwr_ex.c", "stm32f4xx_hal_flash.c",
                    "stm32f4xx_hal_flash_ex.c", "stm32f4xx_hal_exti.c"],
        "board_src": ["board.c", "stm32f4xx_hal_msp.c", "stm32f4xx_it.c"],
        "startup": "startup_stm32f401xe.s", "system": "system_stm32f4xx.c", "ld": "STM32F401RETX_FLASH.ld",
        "lib": "ARMCortexM4/NetworkRuntime1200_CM4_GCC.a",
    },
    "stm32h7b3i_dk": {
        "target": "stm32h7", "defines": ["-DSTM32H7B3xxQ"],
        "cpu": ["-mcpu=cortex-m7", "-mthumb", "-mfpu=fpv5-d16", "-mfloat-abi=hard"],
        "hal": TP / "stm32h7xx-hal-driver", "dev": TP / "cmsis-device-h7",
        "hal_src": ["stm32h7xx_hal.c", "stm32h7xx_hal_cortex.c", "stm32h7xx_hal_rcc.c", "stm32h7xx_hal_rcc_ex.c",
                    "stm32h7xx_hal_gpio.c", "stm32h7xx_hal_uart.c", "stm32h7xx_hal_uart_ex.c", "stm32h7xx_hal_dma.c",
                    "stm32h7xx_hal_dma_ex.c", "stm32h7xx_hal_mdma.c", "stm32h7xx_hal_crc.c", "stm32h7xx_hal_crc_ex.c",
                    "stm32h7xx_hal_pwr.c", "stm32h7xx_hal_pwr_ex.c", "stm32h7xx_hal_flash.c",
                    "stm32h7xx_hal_flash_ex.c", "stm32h7xx_hal_exti.c", "stm32h7xx_hal_hsem.c"],
        "board_src": ["board.c", "stm32h7xx_hal_msp.c", "stm32h7xx_it.c"],
        "startup": "startup_stm32h7b3xxq.s", "system": "system_stm32h7xx.c", "ld": "STM32H7B3LIHXQ_FLASH.ld",
        "lib": "ARMCortexM7/NetworkRuntime1200_CM7_GCC.a",
    },
}
APP_SRC = {"validation": ["common/Src/main_validation.c", "common/Src/app_x-cube-ai.c"],
           "energy": ["common/Src/main_energy.c"], "bnn": ["common/Src/main_bnn.cc"]}
# one-bit firmware: TensorFlow Lite Micro + Larq Compute Engine ops (NAS-BNN's port, fetch_deps.py)
BNN_TP = TP / "tflm_lce"
BNN_SKIP = re.compile(r"_test\.|/test|examples|benchmark|/tools/|python|testing|/signal/|lce_micro/tensorflow|"
                      r"lce_micro/third_party|gemmlowp|ruy/|flatbuffers/src|kissfft")
BNN_INC = [BNN_TP / "TFLM", BNN_TP / "TFLM/third_party/flatbuffers/include", BNN_TP / "TFLM/third_party/gemmlowp",
           BNN_TP / "TFLM/third_party/ruy", BNN_TP / "LCE", BNN_TP / "LCE/lce_micro"]
CXXFLAGS = ["-std=gnu++17", "-O2", "-g", "-ffunction-sections", "-fdata-sections", "-fno-exceptions", "-fno-rtti",
            "-fno-use-cxa-atexit", "-fno-threadsafe-statics", "-DTF_LITE_STATIC_MEMORY", "-DTF_LITE_MCU_DEBUG_LOG",
            "-DNDEBUG", "-DFLATBUFFERS_LOCALE_INDEPENDENT=0", "-DUSE_HAL_DRIVER"]
AI_SRC = {
    "validation": ["Validation/Src/aiValidation.c", "Validation/Src/aiPbMgr.c", "Validation/Src/aiPbIO.c",
                   "Validation/Src/pb_common.c", "Validation/Src/pb_decode.c", "Validation/Src/pb_encode.c",
                   "Validation/Src/stm32msg.pb.c", "Validation/Src/aiPbMemRWServices.c",
                   "Misc/Src/aiTestUtility.c", "Misc/Src/aiTestHelper.c", "Misc/Src/ai_device_adaptor.c",
                   "Misc/Src/lc_print.c", "Misc/Src/syscalls.c"],
    "energy": [],
    "bnn": [],
}
CFLAGS = ["-std=gnu11", "-O2", "-g3", "-ffunction-sections", "-fdata-sections", "-Wall", "-DUSE_HAL_DRIVER"]


def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:], sep="\n")
        raise SystemExit(f"failed: {Path(str(cmd[0])).name} ({r.returncode})")
    return r


def generate(model: Path, out: Path, target: str, mempool: Path = None, optimization: str = None):
    gen = out / "gen"
    if not (gen / "network.c").exists():
        run([STEDGEAI, "generate", "-m", model, "--target", target, "--c-api", "legacy", "--name", "network",
             *(["--memory-pool", mempool] if mempool else []), *(["-O", optimization] if optimization else []),
             "--output", gen, "--workspace", out / "ws"])
    return gen


def title_header(model: Path, gen: Path, out_dir: Path):
    """The first English title of MINDsmall dev: its input (model's own byte table) and the laptop's 8-bit output."""
    import numpy as np
    import onnxruntime as ort
    sys.path.insert(0, str(ROOT))
    from src import board_models, data_mind
    from src.config import load_config
    from src.student import text_to_bytes

    cfg = load_config()
    stem, seed, _ = board_models.split_name(model)
    table = board_models.byte_table(stem, seed)
    dev = data_mind.read_news(cfg, "dev")
    nid = next(n for n, v in dev.items() if v["title"].strip())
    ids = np.array([text_to_bytes(dev[nid]["title"], cfg["data"]["max_title_bytes"])], dtype=np.int64)
    x = np.ascontiguousarray(table[ids].transpose(0, 2, 1)).astype(np.float32)
    m = (ids != 0).astype(np.float32)[:, None, :]
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    y = sess.run(None, {names[0]: x, names[1]: m})[0][0]
    rep = (gen / "network_generate_report.txt").read_text(encoding="utf-8", errors="replace")

    def quant(which):
        """(scale, zero point) of an input or output line of the report, None for a float tensor"""
        line = next(l for l in rep.splitlines() if l.strip().startswith(which))
        s = re.search(r"QLinear\(([\d.e-]+),(-?\d+),int8\)", line)
        return (float(s.group(1)), int(s.group(2))) if s else None

    def q8(v, sz):          # np.round is half to even, as ONNX Runtime's QuantizeLinear
        return np.clip(np.round(v / sz[0]) + sz[1], -128, 127).astype(int)
    scale, zp = quant("output 1/1")
    q = q8(y, (scale, zp))

    def flt(t):
        txt = f"{t:.9g}"
        return txt + ("" if any(c in txt for c in ".eEn") else ".0") + "f"

    def arr(v, fmt):
        return ",\n".join(", ".join(fmt(t) for t in v[i:i + 8]) for i in range(0, len(v), 8))

    def tensor(name, v, which):
        qz = quant(which)
        if qz:              # the network takes 8-bit inputs: the caller quantises (e.g. an 8-bit byte table)
            return f"static const int8_t {name}[{v.size}] = {{\n{arr(q8(v.ravel(), qz), str)}\n}};"
        return f"static const float {name}[{v.size}] = {{\n{arr(v.ravel(), flt)}\n}};"
    out_dir.mkdir(parents=True, exist_ok=True)
    h = ["/* Generated by firmware/build.py: MINDsmall dev news " + nid + " (first English title),",
         " * embedded with the byte table of " + stem + " (seed " + str(seed) + "), quantised to 8 bits where the",
         " * network's input is 8-bit; expected = ONNX Runtime " + ort.__version__ + " output of " + model.name + ",",
         " * quantised with the network's output scale and zero point. */",
         "#include <stdint.h>",
         tensor("title_x", x, "input 1/2"), tensor("title_mask", m, "input 2/2"),
         f"static const int8_t title_expected[{q.size}] = {{\n{arr(q, str)}\n}};", ""]
    (out_dir / "title_input.h").write_text("\n".join(h), encoding="utf-8")
    return nid


def bnn_headers(tflite: Path, out_dir: Path):
    """bnn_model.h (the LCE file) and bnn_titles.h (byte table, test titles' byte ids, LCE interpreter outputs)."""
    import numpy as np
    stem = tflite.stem[:-len("_lce")]
    d = tflite.parent
    P = np.load(d / f"{stem}_params.npz")
    T = np.load(d / f"{stem}_test.npz")
    lce_out = np.load(d / f"{stem}_lce_test_out.npy")
    data = tflite.read_bytes()

    def flt(t):
        txt = f"{t:.9g}"
        return txt + ("" if any(c in txt for c in ".eEn") else ".0") + "f"

    def rows(v, fmt, n=12):
        v = list(v)
        return ",\n".join(", ".join(fmt(t) for t in v[i:i + n]) for i in range(0, len(v), n))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bnn_model.h").write_text(
        f"/* Generated by firmware/build.py from {tflite.name} ({len(data)} B). */\n#include <stdint.h>\n"
        f"alignas(16) static const unsigned char g_bnn_model[{len(data)}] = {{\n{rows(data, str, 24)}\n}};\n"
        f"static const unsigned int g_bnn_model_len = {len(data)};\n", encoding="utf-8")
    ids, n = T["ids"], len(T["ids"])
    h = [f"/* Generated by firmware/build.py: byte table of {stem}, the {n} test titles of scripts/board_validate.py",
         " * (byte ids) and the outputs of LCE's interpreter for them (scripts/lce_convert.py). */",
         "#include <stdint.h>", f"#define N_TITLES {n}", f"#define SEQ_LEN {ids.shape[1]}",
         f"#define EMBED_DIM {P['embed'].shape[1]}", f"#define OUT_DIM {lce_out.shape[1]}",
         f"static const float byte_table[{P['embed'].size}] = {{\n{rows(P['embed'].ravel(), flt)}\n}};",
         f"static const uint16_t title_ids[{n}][{ids.shape[1]}] = {{\n"
         + ",\n".join("{" + rows(r, str, 32) + "}" for r in ids) + "\n};",
         f"static const float lce_out[{n}][{lce_out.shape[1]}] = {{\n"
         + ",\n".join("{" + rows(r, flt) + "}" for r in lce_out) + "\n};", ""]
    (out_dir / "bnn_titles.h").write_text("\n".join(h), encoding="utf-8")


def compile_all(jobs, objs):
    """compile in parallel at below-normal priority; third-party objects newer than their source are kept"""
    from concurrent.futures import ThreadPoolExecutor
    prio = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)

    def one(job):
        cmd, src, obj, always = job
        if not always and obj.exists() and obj.stat().st_mtime > src.stat().st_mtime:
            return job, None
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           creationflags=prio)
        return job, r
    with ThreadPoolExecutor(int(__import__("os").environ.get("FW_JOBS", "3"))) as ex:   # 3: other work shares the laptop
        for job, r in ex.map(one, jobs):
            if r is not None and r.returncode != 0:
                print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
                raise SystemExit(f"failed: {job[1]}")
    return objs


def build(board: str, model: Path, tag: str, app: str, mempool: Path = None, optimization: str = None):
    b = BOARDS[board]
    bdir = HERE / board
    out = bdir / "build" / tag
    objdir = out / app / "obj"
    objdir.mkdir(parents=True, exist_ok=True)
    inc = [bdir / "Core/Inc", HERE / "common/Inc", b["hal"] / "Inc", b["hal"] / "Inc/Legacy", b["dev"] / "Include", CORE]
    extra = {}
    if app == "bnn":
        bnn_headers(model, out / "bnn")
        inc = [out / "bnn", *inc, *BNN_INC]
        gen_c, third = [], [p for p in sorted(BNN_TP.rglob("*")) if p.suffix in (".cc", ".c", ".cpp")
                            and not BNN_SKIP.search(p.relative_to(BNN_TP).as_posix())]
    else:
        gen = generate(model, out, b["target"], mempool, optimization)
        inc = [*inc[:2], gen, *inc[2:], AI / "Inc", AI / "Validation/Inc", AI / "Misc/Inc"]
        gen_c, third = sorted(gen.glob("*.c")), []
        if app == "energy":
            extra["title_nid"] = title_header(model, gen, out / "energy")
            inc.insert(0, out / "energy")
    own = ([HERE / s for s in APP_SRC[app]] + [bdir / "Core/Src" / s for s in b["board_src"]]
           + [AI / s for s in AI_SRC[app]] + gen_c)
    lib_srcs = [b["hal"] / "Src" / s for s in b["hal_src"]] + [b["dev"] / "Source/Templates" / b["system"]] + third
    gcc, gxx = GCC_BIN / "arm-none-eabi-gcc.exe", GCC_BIN / "arm-none-eabi-g++.exe"
    jobs, objs = [], []
    for s, always in [(s, True) for s in own] + [(s, False) for s in lib_srcs]:
        o = objdir / f"{s.stem}_{hashlib.md5(str(s).encode()).hexdigest()[:6]}.o"
        cc, flags = (gxx, CXXFLAGS) if s.suffix in (".cc", ".cpp") else (gcc, CFLAGS)
        jobs.append(([cc, *b["cpu"], *flags, *b["defines"], *[f"-I{i}" for i in inc], "-c", s, "-o", o], s, o, always))
        objs.append(o)
    compile_all(jobs, objs)
    start = objdir / "startup.o"
    run([gcc, *b["cpu"], "-x", "assembler-with-cpp", "-c", b["dev"] / "Source/Templates/gcc" / b["startup"], "-o", start])
    objs.append(start)
    elf = out / app / "firmware.elf"
    run([gxx if app == "bnn" else gcc, *b["cpu"], "-T", bdir / b["ld"], "--specs=nano.specs", "--specs=nosys.specs",
         "-Wl,--gc-sections", f"-Wl,-Map={out / app / 'firmware.map'}", *objs,
         *([] if app == "bnn" else [AI / "Lib/GCC" / b["lib"]]),
         "-Wl,--start-group", "-lc", "-lm", "-Wl,--end-group", "-o", elf])
    size = run([GCC_BIN / "arm-none-eabi-size.exe", "-A", elf]).stdout
    sections = {}
    for line in size.splitlines():
        p = line.split()
        if len(p) >= 3 and p[0].startswith(".") and p[1].isdigit():
            sections[p[0]] = int(p[1])
    flash_b = sum(sections.get(k, 0) for k in (".isr_vector", ".text", ".rodata", ".ARM", ".init_array",
                                                ".fini_array", ".data"))
    ram_b = sum(sections.get(k, 0) for k in (".data", ".bss", "._user_heap_stack", ".axisram"))
    summary = {"board": board, "model": model.name, "tag": tag, "app": app, "elf": str(elf.relative_to(ROOT)),
               "mempool": str(mempool.relative_to(ROOT)) if mempool else None, "optimization": optimization or "balanced",
               "flash_bytes": flash_b, "ram_bytes": ram_b, "sections": sections, **extra,
               "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / app / "build.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("board", "tag", "app", "flash_bytes", "ram_bytes")}))
    return elf


def flash(elf: Path):
    r = run([PROGRAMMER, "-c", "port=SWD", "mode=UR", "-w", elf, "-v", "-rst"])
    print("\n".join(l for l in r.stdout.splitlines() if any(k in l for k in ("Device name", "Download verified",
                                                                         "Error", "reset"))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--board", choices=list(BOARDS), required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--app", choices=["validation", "energy", "bnn"], default="validation")
    ap.add_argument("--mempool", default=None, help="memory-pool file for stedgeai generate (tag gets _mpool)")
    ap.add_argument("--optimization", default=None, help="stedgeai -O (balanced, time, ram); tag gets _O<value>")
    ap.add_argument("--flash", action="store_true")
    a = ap.parse_args()
    model = Path(a.model).resolve()
    mempool = Path(a.mempool).resolve() if a.mempool else None
    tag = a.tag or (model.stem if a.app == "bnn" else model.stem.split("_seed")[0] + ("_mpool" if mempool else "")
                    + (f"_O{a.optimization}" if a.optimization else ""))
    elf = build(a.board, model, tag, a.app, mempool, a.optimization)
    if a.flash:
        flash(elf)
