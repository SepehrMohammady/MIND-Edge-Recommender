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
import os
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

# TensorFlow Lite Micro and the Larq Compute Engine ops for it ("lce_micro"), as ported to the STM32H7B3I-DK by
# the lab's NAS-BNN project (Apache-2.0; NASBNN_DIR points to its STM32/NAS_BNN_Benchmark folder). The source
# tree is pinned by a hash of its files. One fix
# is applied to the copy: the border path of the binary convolution skipped padded taps, which is neither zero
# nor one padding; it now counts them as LCE pads (zero padding: C_in/2 mismatches per padded tap, so that
# K - 2 * accum is the dot product over the valid taps; one padding: the filter's own mismatches against +1).
import hashlib
import shutil

NASBNN = Path(os.environ.get("NASBNN_DIR", TP.parents[2] / "NAS-BNN" / "STM32" / "NAS_BNN_Benchmark"))
PINNED = {"TFLM": "663a0d2f03858482", "LCE": "375d8cf0e9fae5a8"}   # sha256 prefixes of the source trees, 2026-10-19
dst = TP / "tflm_lce"


def tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).replace("\\", "/").encode())
            h.update(p.read_bytes())
    return h.hexdigest()


if not dst.exists():
    for sub in ("TFLM", "LCE"):
        shutil.copytree(NASBNN / sub, dst / sub)
    ref = dst / "LCE/larq_compute_engine/core/bconv2d/reference.h"
    s = ref.read_text(encoding="utf-8")
    old = """          // Slow Path (Edges)
          for (int fy = 0; fy < filter_height; ++fy) {
            const int in_y = in_y_origin + fy * dilation_height_factor;
            if (in_y < 0 || in_y >= input_height) continue;
            for (int fx = 0; fx < filter_width; ++fx) {
              const int in_x = in_x_origin + fx * dilation_width_factor;
              if (in_x < 0 || in_x >= input_width) continue;"""
    new = """          // Slow Path (Edges). Padded taps follow LCE's padding (fix in MIND-Edge-Recommender, firmware/fetch_deps.py):
          // zero padding adds C_in/2 mismatches per padded tap, so that K - 2 * accum is the dot product over the
          // valid taps only; one padding (+1 inputs, bit 0) adds the filter's own mismatches against 0.
          for (int fy = 0; fy < filter_height; ++fy) {
            const int in_y = in_y_origin + fy * dilation_height_factor;
            for (int fx = 0; fx < filter_width; ++fx) {
              const int in_x = in_x_origin + fx * dilation_width_factor;
              if (in_y < 0 || in_y >= input_height || in_x < 0 || in_x >= input_width) {
                for (int oc = 0; oc < output_depth; ++oc) {
                  if (bconv2d_params->pad_value == 0) {
                    accums[oc] += bconv2d_params->channels_in / groups / 2;
                  } else {
                    const TBitpacked* f_ptr = &packed_filter_data[oc * filter_stride + (fy * filter_width + fx) * input_depth_per_group];
                    for (int ic = 0; ic < input_depth_per_group; ++ic) accums[oc] += xor_popcount(0, f_ptr[ic]);
                  }
                }
                continue;
              }"""
    assert s.count(old) == 1, "reference.h border path not found"
    ref.write_text(s.replace(old, new), encoding="utf-8")
    ops = dst / "LCE/lce_micro/lce_ops_micro.cc"
    t = ops.read_text(encoding="utf-8")
    old = """  static bool hack_printed = false;
  if (!hack_printed) {
      printf("LCE Prepare Hack active!\\r\\n");
      hack_printed = true;
  }
"""
    assert t.count(old) == 1, "lce_ops_micro.cc print not found"
    ops.write_text(t.replace(old, ""), encoding="utf-8")
for sub, pin in PINNED.items():
    h = tree_hash(NASBNN / sub)[:16]
    print(f"{'NAS-BNN ' + sub:22s} {'':8s} {h}  {'ok' if h == pin else 'DIFFERENT FROM THE PINNED TREE'}")
