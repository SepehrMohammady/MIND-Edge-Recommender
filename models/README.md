# Model files

The files that ran on the boards and the phone in the paper, all from seed 42.
`python -m scripts.export_release` copies them from the run outputs and writes `manifest.json`
with the size, SHA-256, inputs and outputs, source file and checkpoint of every file.

| Files | Encoder | In the paper |
|---|---|---|
| `unas_h7_seed42_body_{int8qdq,fp32}.onnx` | µNAS 144, recipe choice under the STM32H7B3I-DK budget | Table 3, Figs. 5 and 6a |
| `unas_h7_c140_seed42_body_{int8qdq,fp32}.onnx` | µNAS 140, final choice under the STM32H7B3I-DK budget | Table 3, Figs. 5 and 6 |
| `unas_f401_seed42_body_{int8qdq,fp32}.onnx` | µNAS 134, recipe choice under the NUCLEO-F401RE budget | Table 3, Figs. 5 and 6a |
| `unas_f401_c142_seed42_body_{int8qdq,fp32}.onnx` | µNAS 142, final choice under the NUCLEO-F401RE budget | Table 3, Figs. 5 and 6 |
| `hand_64-2-384_seed42_body_{int8qdq,fp32}.onnx` | hand-designed 64-2-384 | Table 3, Fig. 6a |
| `hand_32-5-384_seed42_body_{int8qdq,fp32}.onnx` | hand-designed 32-5-384 | Table 3, Fig. 6a |
| `hand_64-5-384_seed42_body_{int8qdq,fp32}.onnx` | hand-designed 64-5-384 in the search's terms | Table 3, Fig. 6a |
| `ref_64-5-384_seed42_body_{int8qdq,fp32}.onnx` | reference byte-CNN, distilled start, English clicks | board results of the reference encoder |
| `ref_mixed_64-5-384_seed42_body_{int8qdq,fp32}.onnx` | the same trained on clicks shown in a random xMIND language; the encoder of the phone app | phone results, replay |
| `user_encoder.onnx` | user encoder (additive attention) of the phone app's checkpoint | phone results, replay |
| `reactnet_64-5-384_seed42_lce.tflite` | one-bit encoder with ReActNet blocks, for TensorFlow Lite Micro with the Larq Compute Engine ops | one-bit encoder on the STM32H7B3I-DK |

`*_int8qdq.onnx` is the 8-bit QDQ file (per-channel 8-bit weights and 8-bit inputs to the convolutions
and the output layer); `*_fp32.onnx` is the same network at full precision.

## Inputs

The networks start after the byte-table lookup, which the caller does, as a device would:

1. byte ids: the UTF-8 bytes of the title, the first 128 of them, id = byte + 1, 0 = padding;
2. `embedded_bytes`: rows of the byte table for the ids, as (1, 64, 128) float32. The seven encoders of
   Table 3 share `byte_table_fixed.f32`; the others use `<name>_byte_table.f32`. Each table is
   257 x 64 float32, little-endian;
3. `mask`: 1 where the id is not 0, as (1, 1, 128) float32.

The output `news_embedding` is the 384-d title vector. The click score of a candidate is the dot product
of its vector with the user vector that `user_encoder.onnx` pools from the vectors of the reader's
clicked titles (`history_vectors` (1, H, 384), `history_mask` (1, H)). The one-bit file takes the
embedded bytes as (1, 1, 128, 64) and the mask as (1, 1, 128, 1).

```python
import numpy as np
import onnxruntime as ort

table = np.fromfile("models/byte_table_fixed.f32", dtype="<f4").reshape(257, 64)
title = "..."                                        # a news title, any language
ids = np.zeros(128, np.int64)
raw = title.encode("utf-8")[:128]
ids[:len(raw)] = np.frombuffer(raw, np.uint8) + 1
x = table[ids].T[None].astype(np.float32)            # (1, 64, 128)
mask = (ids != 0).astype(np.float32)[None, None]     # (1, 1, 128)
session = ort.InferenceSession("models/unas_h7_c140_seed42_body_int8qdq.onnx")
vector = session.run(None, {"embedded_bytes": x, "mask": mask})[0]   # (1, 384)
```

## Checkpoints

The PyTorch checkpoints of every run behind Tables 1 to 3 and of the one-bit encoder (115 files) are
packed by `python -m scripts.export_release --checkpoints` into
`release/mind-edge-recommender-checkpoints.zip`, with their SHA-256 in `checkpoints_manifest.json`
inside the archive. The archive goes with the tagged release of this repository, not into git.

## Terms of use

The models were trained on MIND (Microsoft Research License) and xMIND (CC BY-NC-SA 4.0): use them for
non-commercial research only.
