"""Build the NUCLEO-F401RE firmware for one 8-bit encoder, and optionally flash it.

    python firmware/nucleo_f401re/build.py MODEL.onnx [--app validation|energy] [--flash]

  validation  ST's validation application; `stedgeai validate --mode target` then talks to the board
              (scripts/board_validate.py)
  energy      the encoder in a loop on one embedded title, and STOP mode, switched with the blue button
              (Core/Src/main_energy.c); the title, its input and the laptop's 8-bit output are written to
              build/<tag>/energy/title_input.h

Steps: `stedgeai generate` (local ST Edge AI Core 4.0, legacy C API, default options) writes the network
code to build/<tag>/gen; arm-none-eabi-gcc (GNU Tools for STM32 14.3 of STM32CubeIDE 2.1.1) compiles it with
the application, the STM32F4 HAL and CMSIS (firmware/third_party, firmware/fetch_deps.py) and links ST's
network runtime library; STM32_Programmer_CLI flashes the ELF over the board's ST-LINK.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TP = HERE.parent / "third_party"
STEDGEAI_ROOT = Path(r"C:\ST\STEdgeAI\4.0")
STEDGEAI = STEDGEAI_ROOT / "Utilities/windows/stedgeai.exe"
AI = STEDGEAI_ROOT / "Middlewares/ST/AI"
GCC_BIN = Path(r"C:\ST\STM32CubeIDE_2.1.1\STM32CubeIDE\plugins"
               r"\com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.win32_1.0.100.202602081740"
               r"\tools\bin")
PROGRAMMER = Path(r"C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe")

HAL = TP / "stm32f4xx-hal-driver"
DEV = TP / "cmsis-device-f4"
CORE = TP / "CMSIS_5/CMSIS/Core/Include"

HAL_SRC = ["stm32f4xx_hal.c", "stm32f4xx_hal_cortex.c", "stm32f4xx_hal_rcc.c", "stm32f4xx_hal_rcc_ex.c",
           "stm32f4xx_hal_gpio.c", "stm32f4xx_hal_uart.c", "stm32f4xx_hal_dma.c", "stm32f4xx_hal_dma_ex.c",
           "stm32f4xx_hal_crc.c", "stm32f4xx_hal_pwr.c", "stm32f4xx_hal_pwr_ex.c", "stm32f4xx_hal_flash.c",
           "stm32f4xx_hal_flash_ex.c", "stm32f4xx_hal_exti.c"]
APP_SRC = {
    "validation": ["Core/Src/main.c", "Core/Src/app_x-cube-ai.c", "Core/Src/board.c", "Core/Src/stm32f4xx_hal_msp.c",
                   "Core/Src/stm32f4xx_it.c"],
    "energy": ["Core/Src/main_energy.c", "Core/Src/board.c", "Core/Src/stm32f4xx_hal_msp.c", "Core/Src/stm32f4xx_it.c"],
}
AI_SRC = {
    "validation": ["Validation/Src/aiValidation.c", "Validation/Src/aiPbMgr.c", "Validation/Src/aiPbIO.c",
                   "Validation/Src/pb_common.c", "Validation/Src/pb_decode.c", "Validation/Src/pb_encode.c",
                   "Validation/Src/stm32msg.pb.c", "Validation/Src/aiPbMemRWServices.c",
                   "Misc/Src/aiTestUtility.c", "Misc/Src/aiTestHelper.c", "Misc/Src/ai_device_adaptor.c",
                   "Misc/Src/lc_print.c", "Misc/Src/syscalls.c"],
    "energy": [],
}

CPU = ["-mcpu=cortex-m4", "-mthumb", "-mfpu=fpv4-sp-d16", "-mfloat-abi=hard"]
CFLAGS = CPU + ["-std=gnu11", "-O2", "-g3", "-ffunction-sections", "-fdata-sections", "-Wall",
                "-DUSE_HAL_DRIVER", "-DSTM32F401xE"]


def run(cmd, **kw):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace", **kw)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:], sep="\n")
        raise SystemExit(f"failed: {Path(str(cmd[0])).name} ({r.returncode})")
    return r


def generate(model: Path, out: Path):
    gen = out / "gen"
    if not (gen / "network.c").exists():
        run([STEDGEAI, "generate", "-m", model, "--target", "stm32f4", "--c-api", "legacy", "--name", "network",
             "--output", gen, "--workspace", out / "ws"])
    return gen


def title_header(model: Path, gen: Path, out_dir: Path, seed: int = 42):
    """The first English title of MINDsmall dev: its input (model's own byte table) and the laptop's 8-bit output."""
    import re
    import numpy as np
    import onnxruntime as ort
    import torch
    sys.path.insert(0, str(ROOT))
    from src import data_mind
    from src.config import load_config
    from src.student import text_to_bytes

    cfg = load_config()
    stem = model.stem.split("_seed")[0]                                  # unas_f401_c142
    search = "mind_" + stem[len("unas_"):]
    sd = torch.load(ROOT / f"artifacts/runs/unas_full/{search}_distill_ft_en_seed{seed}.pt", map_location="cpu")
    table = sd["news_encoder.embed.weight"].numpy().astype(np.float32)
    dev = data_mind.read_news(cfg, "dev")
    nid = next(n for n, v in dev.items() if v["title"].strip())
    ids = np.array([text_to_bytes(dev[nid]["title"], cfg["data"]["max_title_bytes"])], dtype=np.int64)
    x = np.ascontiguousarray(table[ids].transpose(0, 2, 1)).astype(np.float32)
    m = (ids != 0).astype(np.float32)[:, None, :]
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    y = sess.run(None, {names[0]: x, names[1]: m})[0][0]
    rep = (gen / "network_generate_report.txt").read_text(encoding="utf-8", errors="replace")
    s = re.search(r"QLinear\(([\d.e-]+),(-?\d+),int8\)", rep)
    scale, zp = float(s.group(1)), int(s.group(2))
    q = np.clip(np.round(y / scale + zp), -128, 127).astype(int)

    def flt(t):
        txt = f"{t:.9g}"
        return txt + ("" if any(c in txt for c in ".eEn") else ".0") + "f"

    def arr(v, fmt):
        return ",\n".join(", ".join(fmt(t) for t in v[i:i + 8]) for i in range(0, len(v), 8))
    out_dir.mkdir(parents=True, exist_ok=True)
    h = ["/* Generated by firmware/nucleo_f401re/build.py: MINDsmall dev news " + nid + " (first English title),",
         " * embedded with the byte table of " + stem + " (seed " + str(seed) + "); expected = ONNX Runtime " + ort.__version__,
         " * output of " + model.name + ", quantised with the network's output scale and zero point. */",
         "#include <stdint.h>",
         f"static const float title_x[{x.size}] = {{\n{arr(x.ravel(), flt)}\n}};",
         f"static const float title_mask[{m.size}] = {{\n{arr(m.ravel(), flt)}\n}};",
         f"static const int8_t title_expected[{q.size}] = {{\n{arr(q, str)}\n}};", ""]
    (out_dir / "title_input.h").write_text("\n".join(h), encoding="utf-8")
    return nid


def build(model: Path, tag: str, app: str):
    out = HERE / "build" / tag
    gen = generate(model, out)
    objdir = out / app / "obj"
    objdir.mkdir(parents=True, exist_ok=True)
    inc = [HERE / "Core/Inc", gen, HAL / "Inc", HAL / "Inc/Legacy", DEV / "Include", CORE,
           AI / "Inc", AI / "Validation/Inc", AI / "Misc/Inc"]
    extra = {}
    if app == "energy":
        extra["title_nid"] = title_header(model, gen, out / "energy")
        inc.insert(0, out / "energy")
    srcs = ([HERE / s for s in APP_SRC[app]] + [HAL / "Src" / s for s in HAL_SRC] + [AI / s for s in AI_SRC[app]]
            + sorted(gen.glob("*.c")) + [DEV / "Source/Templates/system_stm32f4xx.c"])
    gcc = GCC_BIN / "arm-none-eabi-gcc.exe"
    objs = []
    for s in srcs:
        o = objdir / (s.stem + ".o")
        run([gcc, *CFLAGS, *[f"-I{i}" for i in inc], "-c", s, "-o", o])
        objs.append(o)
    start = objdir / "startup_stm32f401xe.o"
    run([gcc, *CPU, "-x", "assembler-with-cpp", "-c", DEV / "Source/Templates/gcc/startup_stm32f401xe.s", "-o", start])
    objs.append(start)
    elf = out / app / "firmware.elf"
    lib = AI / "Lib/GCC/ARMCortexM4/NetworkRuntime1200_CM4_GCC.a"
    run([gcc, *CPU, "-T", HERE / "STM32F401RETX_FLASH.ld", "--specs=nano.specs", "--specs=nosys.specs",
         "-Wl,--gc-sections", f"-Wl,-Map={out / app / 'firmware.map'}",
         *objs, lib, "-Wl,--start-group", "-lc", "-lm", "-Wl,--end-group", "-o", elf])
    size = run([GCC_BIN / "arm-none-eabi-size.exe", "-A", elf]).stdout
    sections = {}
    for line in size.splitlines():
        p = line.split()
        if len(p) >= 3 and p[0].startswith(".") and p[1].isdigit():
            sections[p[0]] = int(p[1])
    flash_b = sum(sections.get(k, 0) for k in (".isr_vector", ".text", ".rodata", ".ARM", ".init_array",
                                                ".fini_array", ".data"))
    ram_b = sum(sections.get(k, 0) for k in (".data", ".bss", "._user_heap_stack"))
    summary = {"model": model.name, "tag": tag, "app": app, "elf": str(elf.relative_to(ROOT)), "flash_bytes": flash_b,
               "ram_bytes": ram_b, "sections": sections, **extra, "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / app / "build.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("tag", "app", "flash_bytes", "ram_bytes")}))
    return elf


def flash(elf: Path):
    r = run([PROGRAMMER, "-c", "port=SWD", "mode=UR", "-w", elf, "-v", "-rst"])
    print("\n".join(l for l in r.stdout.splitlines() if any(k in l for k in ("Device name", "Download verified",
                                                                         "Error", "reset"))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--app", choices=["validation", "energy"], default="validation")
    ap.add_argument("--flash", action="store_true")
    a = ap.parse_args()
    model = Path(a.model).resolve()
    tag = a.tag or model.stem.split("_seed")[0].replace("unas_f401_", "")
    elf = build(model, tag, a.app)
    if a.flash:
        flash(elf)
