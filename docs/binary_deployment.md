# Binary deployment path (Linux / ARM)

The improved binary encoder (`src/binary.py`, ReActNet + Bi-Real) reaches
**0.572 AUC at ~186 KB** on MINDsmall-dev (up from 0.521 naive). Real 1-bit
execution needs an ARM binary runtime, which is **not available on Windows**, so
this is the procedure for when a Linux box / WSL2 and the boards are available.

## Why it can't run on the dev laptop (verified)
- `larq-compute-engine` (LCE — the 1-bit converter/runtime) has **no Windows
  wheel** (Linux/macOS only).
- `larq` 0.14 is archived and needs **Keras 2** (works only via the `tf-keras`
  shim with `TF_USE_LEGACY_KERAS=1`) and **numpy < 2**.
- TensorFlow 2.21 itself installs fine on Windows/Python 3.13 — only LCE is the blocker.

## Procedure (Linux or WSL2; Python 3.10–3.11 recommended for old Larq)
1. `python -m venv .venv-lce && .venv-lce/bin/pip install "tensorflow==2.15.*" "numpy<2" larq larq-compute-engine`
2. Rebuild the encoder in Larq/Keras, matching `src/binary.py` shapes: byte
   embedding (FP) → 1×1 proj (FP) → N × [`larq.layers.QuantConv1D`(kernel &
   input quantizer = `ste_sign`) + BatchNorm + RPReLU + residual] → dense head (FP).
   Port the trained weights, or re-distill to the cached teacher anchors.
3. Convert to a bit-packed binary TFLite:
   ```python
   import larq_compute_engine as lce
   open("encoder_binary.tflite", "wb").write(lce.convert_keras_model(model))
   ```
   → real bit-packed size; benchmark with the LCE runtime.
4. **Raspberry Pi 5 (aarch64):** run the `.tflite` with the LCE runtime; measure
   latency + power (INA219 on the 5 V rail).
5. **STM32H7:** import the Larq/QKeras model into ST X-CUBE-AI (deeply-quantized
   1-bit path) for flash / RAM / cycle reports.

Until a board + Linux are available, binary footprint/energy are reported
**analytically** in the paper; **INT8** (ONNX → X-CUBE-AI) is the measured MCU path.
