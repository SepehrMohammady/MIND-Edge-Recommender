"""Step 2 of scripts/export_binary.py, run in WSL (TensorFlow 2.13, larq 0.13.3, larq-compute-engine 0.16.0):

    nice -n 10 ~/lce_env/bin/python scripts/lce_convert.py [--seed 42]      # from the repository root

Rebuilds the one-bit ReActNet encoder in Larq from artifacts/binary_lce/<name>_params.npz, NHWC with height 1:
embedded bytes (1, 1, 128, 64) and the padding mask (1, 1, 128, 1) in, the 384-d vector out (the byte-table
lookup is the caller's, as for the 8-bit files). Per block: x - threshold, QuantConv2D (1x3, ste_sign inputs and
kernel, zero padding as in PyTorch), * mult + add, RPReLU (- b1, PReLU, + b2), + shortcut. Checks the Larq
model against the PyTorch outputs of the 20 test titles, converts it with larq_compute_engine.convert_keras_model,
runs LCE's interpreter on the test titles and on the dev vocabulary, and writes
  artifacts/binary_lce/<name>_lce.tflite, <name>_lce_dev_vectors.npy, <name>_lce_test_out.npy, <name>_lce_convert.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import larq as lq
import larq_compute_engine as lce

ROOT = Path(__file__).resolve().parents[1]
L, E, C, D = 128, 64, 64, 384


def build(P):
    def shift(v, name):
        c = tf.constant(v.reshape(-1).astype(np.float32))       # (C,): broadcasts over the channel axis
        return tf.keras.layers.Lambda(lambda t: t + c, name=name)

    x_in = tf.keras.Input((1, L, E), batch_size=1, name="embedded_bytes")
    m_in = tf.keras.Input((1, L, 1), batch_size=1, name="mask")
    proj = tf.keras.layers.Conv2D(C, (1, 1), name="proj_in")
    x = proj(x_in)
    layers = {"proj_in": proj}
    for i in range(5):
        s = shift(-P[f"b{i}_thr"], f"b{i}_thr")(x)
        conv = lq.layers.QuantConv2D(C, (1, 3), padding="same", pad_values=0.0, input_quantizer="ste_sign",
                                     kernel_quantizer="ste_sign", kernel_constraint=None, use_bias=False,
                                     name=f"b{i}_bconv")
        y = conv(s)
        mult = tf.constant(P[f"b{i}_mult"].reshape(-1))         # 1-D, so LCE fuses it into the binary conv
        add = tf.constant(P[f"b{i}_add"].reshape(-1))
        y = tf.keras.layers.Lambda(lambda t, m=mult, a=add: t * m + a, name=f"b{i}_affine")(y)
        y = shift(-P[f"b{i}_rp_b1"], f"b{i}_rp_b1")(y)
        pr = tf.keras.layers.PReLU(shared_axes=[1, 2], name=f"b{i}_prelu")
        y = pr(y)
        y = shift(P[f"b{i}_rp_b2"], f"b{i}_rp_b2")(y)
        x = tf.keras.layers.Add(name=f"b{i}_shortcut")([y, x])
        layers[f"b{i}_bconv"], layers[f"b{i}_prelu"] = conv, pr
    xs = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=[1, 2]), name="masked_sum")([x, m_in])
    ms = tf.keras.layers.Lambda(lambda t: tf.maximum(tf.reduce_sum(t, axis=[1, 2]), 1.0), name="mask_count")(m_in)
    pooled = tf.keras.layers.Lambda(lambda t: t[0] / t[1], name="masked_mean")([xs, ms])
    head = tf.keras.layers.Dense(D, name="head")
    out = head(pooled)
    layers["head"] = head
    model = tf.keras.Model([x_in, m_in], out)

    proj.set_weights([P["proj_w"].T.reshape(1, 1, E, C), P["proj_b"]])
    for i in range(5):
        w = P[f"b{i}_wsign"]                                           # (out, in, k)
        layers[f"b{i}_bconv"].set_weights([np.transpose(w, (2, 1, 0))[None]])   # (1, k, in, out)
        layers[f"b{i}_prelu"].set_weights([P[f"b{i}_rp_a"].reshape(1, 1, C)])
    head.set_weights([P["head_w"].T, P["head_b"]])
    return model


def inputs(table, ids):
    x = table[ids][:, None, :, :].astype(np.float32)                   # (N, 1, 128, 64)
    m = (ids != 0).astype(np.float32)[:, None, :, None]                # (N, 1, 128, 1)
    return x, m


def cosines(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    name = f"reactnet_64-5-384_seed{a.seed}"
    d = ROOT / "artifacts/binary_lce"
    P = dict(np.load(d / f"{name}_params.npz"))
    T = np.load(d / f"{name}_test.npz")
    t0 = time.time()
    model = build(P)
    x, m = inputs(P["embed"], T["ids"])
    keras_out = np.concatenate([model([x[i:i + 1], m[i:i + 1]], training=False).numpy() for i in range(len(x))])
    ref = T["torch_out"]
    res = {"model": name, "tensorflow": tf.__version__, "larq": lq.__version__, "lce": lce.__version__,
           "keras_vs_torch": {"max_abs_diff": float(np.abs(keras_out - ref).max()),
                              "min_cosine": float(cosines(keras_out, ref).min()), "titles": int(len(x))}}
    flat = lce.convert_keras_model(model)
    path = d / f"{name}_lce.tflite"
    path.write_bytes(flat)
    res["tflite_bytes"] = len(flat)
    itp = lce.testing.Interpreter(flat, num_threads=1)
    lce_out = np.concatenate([np.asarray(itp.predict([x[i:i + 1], m[i:i + 1]])).reshape(1, -1) for i in range(len(x))])
    np.save(d / f"{name}_lce_test_out.npy", lce_out.astype(np.float32))
    res["lce_vs_keras"] = {"max_abs_diff": float(np.abs(lce_out - keras_out).max()),
                           "min_cosine": float(cosines(lce_out, keras_out).min())}
    res["lce_vs_torch"] = {"max_abs_diff": float(np.abs(lce_out - ref).max()), "min_cosine": float(cosines(lce_out, ref).min())}
    print(json.dumps(res, indent=1), flush=True)

    dev = np.load(d / f"{name}_dev_ids.npy")
    vec = np.zeros((len(dev), D), dtype=np.float32)
    t1 = time.time()
    for i in range(len(dev)):
        if dev[i].any():
            xi, mi = inputs(P["embed"], dev[i:i + 1])
            vec[i] = np.asarray(itp.predict([xi, mi])).reshape(-1)
    np.save(d / f"{name}_lce_dev_vectors.npy", vec)
    res["dev_titles"] = int(len(dev))
    res["lce_interpreter_ms_per_title_x86"] = round(1000 * (time.time() - t1) / len(dev), 3)
    res["minutes"] = round((time.time() - t0) / 60, 2)
    res["ops"] = sorted({op for op in _ops(flat)})
    (d / f"{name}_lce_convert.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("dev_titles", "lce_interpreter_ms_per_title_x86", "minutes", "ops")}, indent=1))


def _ops(flat):
    """operator names in the flatbuffer (builtin and custom), via TensorFlow's analyzer output"""
    try:
        from tensorflow.lite.python import schema_py_generated as schema
        m = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(flat, 0))
        names = []
        for oc in m.operatorCodes:
            if oc.customCode:
                names.append(oc.customCode.decode() if isinstance(oc.customCode, bytes) else str(oc.customCode))
            else:
                code = max(oc.builtinCode, oc.deprecatedBuiltinCode)
                names.append({v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}.get(code, str(code)))
        return names
    except Exception as e:  # noqa: BLE001
        return [f"unknown ({e})"]


if __name__ == "__main__":
    main()
