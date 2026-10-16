"""Fetch the third-party sources of the board firmware into firmware/third_party (not committed).

    python firmware/fetch_deps.py

Pinned versions (fetched 2026-10-15):
  cmsis-device-f4       v2.6.11  0fa0e489e053fa1ca7790bb40b4d76458f64c55d  (Apache-2.0)
  stm32f4xx-hal-driver  v1.8.5   b6f0ed3829f3829eb358a2e7417d80bba1a42db7  (BSD-3-Clause)
  CMSIS_5 (Core/Include) 5.9.0   2b7495b8535bdcb306dac29b9ded4cfb679d7e5c  (Apache-2.0)
  cmsis-device-h7       v1.10.7  8f922cdc7cc6de2344e75ddd657889f4ff761790  (Apache-2.0, fetched 2026-10-16)
  stm32h7xx-hal-driver  v1.11.6  a1996eed9172b59887bafaaa0ea1816ea14d48b5  (BSD-3-Clause, fetched 2026-10-16)

ST Edge AI Core 4.0 (network code generator, runtime library and validation application) and the GNU
toolchain of STM32CubeIDE are local installs; see firmware/README.md.
"""
import subprocess
from pathlib import Path

TP = Path(__file__).resolve().parent / "third_party"
REPOS = [
    ("cmsis-device-f4", "https://github.com/STMicroelectronics/cmsis-device-f4.git", "v2.6.11",
     "0fa0e489e053fa1ca7790bb40b4d76458f64c55d", None),
    ("stm32f4xx-hal-driver", "https://github.com/STMicroelectronics/stm32f4xx-hal-driver.git", "v1.8.5",
     "b6f0ed3829f3829eb358a2e7417d80bba1a42db7", None),
    ("CMSIS_5", "https://github.com/ARM-software/CMSIS_5.git", "5.9.0",
     "2b7495b8535bdcb306dac29b9ded4cfb679d7e5c", "CMSIS/Core/Include"),
    ("cmsis-device-h7", "https://github.com/STMicroelectronics/cmsis-device-h7.git", "v1.10.7",
     "8f922cdc7cc6de2344e75ddd657889f4ff761790", None),
    ("stm32h7xx-hal-driver", "https://github.com/STMicroelectronics/stm32h7xx-hal-driver.git", "v1.11.6",
     "a1996eed9172b59887bafaaa0ea1816ea14d48b5", None),
]

TP.mkdir(exist_ok=True)
for name, url, tag, commit, sparse in REPOS:
    d = TP / name
    if not d.exists():
        cmd = ["git", "clone", "-q", "--depth", "1", "--branch", tag]
        if sparse:
            cmd += ["--filter=blob:none", "--sparse"]
        subprocess.run(cmd + [url, str(d)], check=True)
        if sparse:
            subprocess.run(["git", "-C", str(d), "sparse-checkout", "set", sparse], check=True)
    head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"{name:22s} {tag:8s} {head}  {'ok' if head == commit else 'DIFFERENT FROM THE PINNED COMMIT'}")
