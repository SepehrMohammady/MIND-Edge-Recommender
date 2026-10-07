"""Deployment files for the boards and the accuracy of the integer model.

1. Export the trained 64-5-384 encoder (October programme, distilled start, English
   clicks, seed 42) to ONNX with a fixed batch of one.
2. Quantise it to an 8-bit QDQ ONNX model with ONNX Runtime static quantisation
   (per-channel symmetric int8 weights, int8 activations), calibrated on 512
   training titles; also quantise the two other searched architectures (quick-run
   weights; their latency and footprint do not depend on the weights).
3. Measure the ranking quality of the exported files on the 73,152 development
   impressions: news vectors from ONNX Runtime, the saved user encoder on top.

Runs under an interpreter whose torch loads (the DIMIR venv on this laptop):
    PYTHONPATH=C:/Projects/PhD/MIND C:/Projects/PhD/DIMIR/.venv/Scripts/python.exe -m scripts.deploy_int8
Writes paper/results/deploy_int8.json and artifacts/stedgeai/models/*.onnx.
"""
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from src import data_mind, recommender
from src.config import load_config
from src.student import ByteCNNEncoder, _byte_matrix

cfg = load_config()
L = cfg["data"]["max_title_bytes"]
OUT = Path("artifacts/stedgeai/models")
OUT.mkdir(parents=True, exist_ok=True)
RES = Path(cfg["paths"]["results_dir"]) / "deploy_int8.json"
CKPT = Path("artifacts/runs/p1/distill_ft_en_seed42.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
t_start = time.time()


def export_static(encoder, path):
    encoder.eval()
    dummy = torch.zeros(1, L, dtype=torch.long)
    torch.onnx.export(encoder, (dummy,), str(path), input_names=["title_bytes"], output_names=["news_embedding"],
                      opset_version=18, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))


def int32_variant(src, dst):
    m = onnx.load(str(src))
    inp = m.graph.input[0]
    old = inp.name
    inp.name = old + "_i32"
    inp.type.tensor_type.elem_type = TensorProto.INT32
    m.graph.node.insert(0, helper.make_node("Cast", [inp.name], [old], to=TensorProto.INT64, name="cast_ids"))
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


class Titles(CalibrationDataReader):
    def __init__(self, mat, input_name):
        self.mat, self.name, self.i = mat, input_name, 0

    def get_next(self):
        if self.i >= len(self.mat):
            return None
        x = self.mat[self.i:self.i + 1].astype(np.int64)
        self.i += 1
        return {self.name: x}


def quantise(fp32_path, int8_path, calib):
    pre = fp32_path.with_name(fp32_path.stem + "_pre.onnx")
    quant_pre_process(str(fp32_path), str(pre))
    name = ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    quantize_static(str(pre), str(int8_path), Titles(calib, name), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv", "Gemm", "MatMul"])
    pre.unlink()


def ort_vectors(path, byte_matrix):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    dtype = np.int32 if "int32" in sess.get_inputs()[0].type else np.int64
    out = np.zeros((len(byte_matrix), 384), dtype=np.float32)
    mat = byte_matrix.numpy()
    for i in range(len(mat)):
        if mat[i].any():
            out[i] = sess.run(None, {name: mat[i:i + 1].astype(dtype)})[0][0]
    return torch.tensor(np.nan_to_num(out))


# ---------------------------------------------------------------- trained encoder and recommender
sd = torch.load(CKPT, map_location="cpu")
enc = ByteCNNEncoder(byte_embed_dim=64, channels=64, depth=5, out_dim=384)
enc.load_state_dict({k[len("news_encoder."):]: v for k, v in sd.items() if k.startswith("news_encoder.")})
fp32 = OUT / "micro_nas_64-5-384_distilled_fp32_b1.onnx"
export_static(enc, fp32)                      # exported on the CPU before the recommender moves to the GPU
int8 = OUT / "micro_nas_64-5-384_distilled_int8qdq_b1.onnx"
rec_torch = recommender.NewsRecommender(enc).to(device)
rec_torch.load_state_dict(sd)
rec_torch.eval()

# calibration titles: 512 training titles (English)
train_news = data_mind.read_news(cfg, "train")
titles = [v["title"] for v in list(train_news.values())[:512]]
calib = _byte_matrix(titles, L)
quantise(fp32, int8, calib)
for p in (fp32, int8):
    int32_variant(p, p.with_name(p.stem + "_i32.onnx"))

# the other two searched architectures (quick-run weights): int8 files for footprint and latency
for stem in ("content_encoder_nas_fp32_b1", "content_encoder_bin_unas_fp32_b1"):
    src = OUT / f"{stem}.onnx"
    if src.exists():
        dst = OUT / f"{stem.replace('_fp32_b1', '')}_int8qdq_b1.onnx"
        quantise(src, dst, calib)
        int32_variant(dst, dst.with_name(dst.stem + "_i32.onnx"))

# ---------------------------------------------------------------- accuracy on the development impressions
vocab = recommender.eval_vocab(cfg, "dev")
results = {"checkpoint": str(CKPT), "calibration_titles": len(titles), "files": {}}
with torch.no_grad():
    results["torch_fp32"] = recommender.evaluate(cfg, rec_torch, "dev")
print("torch fp32", results["torch_fp32"])

user_only = recommender.NewsRecommender(recommender.FixedVectors(384)).to(device)
missing = user_only.load_state_dict({k: v for k, v in sd.items() if not k.startswith("news_encoder.")}, strict=False)
print("user encoder loaded; missing keys:", missing.missing_keys)
vec = {}
for key, path in (("onnx_fp32", fp32), ("onnx_int8qdq", int8)):
    t0 = time.time()
    V = ort_vectors(path, vocab.byte_matrix)
    vec[key] = V
    with torch.no_grad():
        results[key] = recommender.evaluate(cfg, user_only, "dev", news_vectors=V)
    results[key]["encode_minutes"] = round((time.time() - t0) / 60, 2)
    results["files"][key] = {"path": str(path), "bytes": path.stat().st_size}
    print(key, results[key])
cos = torch.nn.functional.cosine_similarity(vec["onnx_fp32"][1:], vec["onnx_int8qdq"][1:], dim=-1)
results["cosine_fp32_vs_int8"] = {"mean": float(cos.mean()), "min": float(cos.min()), "p05": float(cos.quantile(0.05))}
results["minutes"] = round((time.time() - t_start) / 60, 2)
results["torch"] = torch.__version__
results["onnxruntime"] = ort.__version__
RES.write_text(json.dumps(results, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in results.items() if k != "files"}, indent=1))
