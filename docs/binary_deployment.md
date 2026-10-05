# Binary deployment path

Status 2026-10-05: nothing binary has run on a board yet. This note records the
routes that exist and what each needs.

The ReActNet / Bi-Real encoder (`src/binary.py`, architecture 64-5-384) reached
**0.572 AUC** on MINDsmall dev in June 2026 (naive weight binarisation: 0.521).
Its estimated size is **194 KB**: 61,440 one-bit weights plus 47,808
full-precision parameters (byte embedding, first projection, output layer, batch
normalisation, RSign / RPReLU). It has 8.41 M operations per title, 7.86 M of
them one-bit.

## Routes to a microcontroller

1. **Larq Compute Engine on TensorFlow Lite Micro.** The route of the lab's
   WakeVision / COINS 2026 work on the STM32H7B3I-DK (the
   `nasbnn-wakevision-stm32` code): rebuild the
   network in Keras with Larq layers, port the PyTorch weights, convert with
   `larq_compute_engine.convert_keras_model`, run with the LCE operators inside
   TFLM. The byte encoder is 1D; LCE's binary convolution is 2D, so each block
   becomes a 1×3 `QuantConv2D`. Known cost from that work: the LCE `BConv2D`
   kernel asks for a scratch buffer that depends on the channel count.
2. **CBin-NN** (Sakr et al., Electronics 13(9):1624, 2024): the lab's C
   inference engine for binarized
   networks, with operators for float-in/binary-out, binary/binary and
   binary/float-out layers and generated C headers. No TensorFlow dependency on
   the board. Needs 1D operators or the same 1×k mapping.
3. **ST Edge AI (X-CUBE-AI) deeply quantized import.** Accepts 1-bit layers only
   from Keras models built with QKeras or Larq. ST Edge AI Core 4.0.0 marks the
   Larq layers as deprecated, to be removed in the next release, so this route
   should not be the plan. Its documentation lists no ONNX route to 1-bit kernels.

For 8-bit models none of this is needed: quantized ONNX (QDQ) or INT8 TFLite goes
through ST Edge AI, which is how the Lane-Change-MCU project measures its models
on the STM32H7B3I-DK and NUCLEO-F401RE.

## What a binary measurement needs

- A Keras/Larq (or CBin-NN) rebuild of `BinaryByteCNNEncoder` with the trained
  weights, and a check that its outputs match the PyTorch model on the dev titles.
- The byte-embedding lookup either inside the graph (Gather) or done in C before
  the network, feeding a (128, 64) tensor.
- Latency, flash and RAM from the board; energy with the lab power probe.
- The Cortex-M7 has no popcount instruction, so a 1-bit kernel counts bits in
  software. Until measured, binary is treated as a way to save flash, with no
  speed claim.

## Windows note

`larq-compute-engine` has no Windows wheel and `larq` 0.14 needs Keras 2
(`tf-keras` with `TF_USE_LEGACY_KERAS=1`) and numpy < 2. The conversion step runs
under Linux or WSL2 (Python 3.10–3.11).
