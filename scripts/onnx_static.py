"""Make deployment copies of the exported encoders without touching PyTorch: fix the
batch dimension to 1 and, in a second variant, take the byte ids as int32 through
a Cast node (ST Edge AI accepts int32 inputs more readily than int64). The
weights are untouched, so size, operations and latency of the architecture are
unchanged; the copies are verified against the original with ONNX Runtime.

    python -m scripts.onnx_static artifacts/runs/quick/content_encoder_micro_nas_fp32.onnx ...
"""
import pathlib
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

OUT = pathlib.Path("artifacts/stedgeai/models")
OUT.mkdir(parents=True, exist_ok=True)


def static_batch(model):
    for t in list(model.graph.input) + list(model.graph.output):
        d = t.type.tensor_type.shape.dim[0]
        d.ClearField("dim_param")
        d.dim_value = 1
    for vi in model.graph.value_info:
        if vi.type.tensor_type.shape.dim and vi.type.tensor_type.shape.dim[0].dim_param == "batch":
            vi.type.tensor_type.shape.dim[0].ClearField("dim_param")
            vi.type.tensor_type.shape.dim[0].dim_value = 1
    return model


def int32_input(model):
    inp = model.graph.input[0]
    old = inp.name
    inp.name = old + "_i32"
    inp.type.tensor_type.elem_type = TensorProto.INT32
    cast = helper.make_node("Cast", [inp.name], [old], to=TensorProto.INT64, name="cast_ids")
    model.graph.node.insert(0, cast)
    return model


def check(orig_path, new_path, i32):
    L = 128
    x = np.random.randint(1, 257, size=(1, L)).astype(np.int64)
    x[0, 90:] = 0
    a = ort.InferenceSession(str(orig_path), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(new_path), providers=["CPUExecutionProvider"])
    ya = a.run(None, {a.get_inputs()[0].name: x})[0]
    yb = b.run(None, {b.get_inputs()[0].name: x.astype(np.int32) if i32 else x})[0]
    return float(np.abs(ya - yb).max())


for src in sys.argv[1:]:
    src = pathlib.Path(src)
    m = onnx.load(str(src))
    m = static_batch(m)
    onnx.checker.check_model(m)
    p1 = OUT / (src.stem + "_b1.onnx")
    onnx.save(m, str(p1), save_as_external_data=False)
    m2 = int32_input(onnx.load(str(p1)))
    onnx.checker.check_model(m2)
    p2 = OUT / (src.stem + "_b1_i32.onnx")
    onnx.save(m2, str(p2), save_as_external_data=False)
    print(src.name, "->", p1.name, "maxdiff", check(src, p1, False), "|", p2.name, "maxdiff", check(src, p2, True),
          "| bytes", p1.stat().st_size, p2.stat().st_size)
