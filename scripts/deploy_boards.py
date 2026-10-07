"""Deployment files for the ST boards, and the accuracy of the deployed integer file.

The network is exported without its byte-table lookup: the caller gathers the 128
rows of the 257 x 64 table (65,792 bytes, kept with the application) and hands the
network the embedded sequence (1, 64, 128) and the padding mask (1, 1, 128). This
is the usual arrangement for an embedding lookup on a microcontroller, and it lets
the board benchmark of the ST Edge AI Developer Cloud, which validates each model
on the target with random inputs in (0, 1), check the whole network numerically.
A first variant with the lookup inside the graph ran on the boards but failed that
check (relative L2 error 0.43 at full precision), so the lookup was moved out.

1. 64-5-384, trained weights (October programme, distilled start, English clicks,
   seed 42): full-precision ONNX and 8-bit QDQ ONNX (ONNX Runtime static
   quantisation, per-channel symmetric int8 weights, int8 activations, 512 training
   titles for calibration), plus the AUC of both files on the 73,152 development
   impressions through the saved user encoder.
2. 256-4-384 and 96-2-384 with random weights: the same two files each, for
   footprint and latency only (both are independent of the weight values).

Runs under an interpreter whose torch loads (the DIMIR venv on this laptop):
    PYTHONPATH=C:/Projects/PhD/MIND C:/Projects/PhD/DIMIR/.venv/Scripts/python.exe -m scripts.deploy_boards
Writes artifacts/stedgeai/models/<arch>_body_{fp32,int8qdq}.onnx and paper/results/deploy_int8.json.
"""
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
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


class Body(torch.nn.Module):
    """The encoder after the byte-table lookup: embedded sequence and mask in, news vector out."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, x, mask):                  # x (1, E, L) float32, mask (1, 1, L) float32
        x = self.enc.proj_in(x)
        for blk in self.enc.blocks:
            x = blk(x)
        x = (x * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return self.enc.head(x)


def embed_inputs(table, ids):
    """numpy lookup: ids (N, L) int -> x (N, E, L) float32, mask (N, 1, L) float32."""
    x = np.ascontiguousarray(table[ids].transpose(0, 2, 1)).astype(np.float32)
    mask = (ids != 0).astype(np.float32)[:, None, :]
    return x, mask


def export(enc, path):
    enc.eval()
    E = enc.embed.weight.shape[1]
    torch.onnx.export(Body(enc), (torch.zeros(1, E, L), torch.ones(1, 1, L)), str(path),
                      input_names=["embedded_bytes", "mask"], output_names=["news_embedding"], opset_version=18, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))


class Calib(CalibrationDataReader):
    def __init__(self, x, mask, names):
        self.x, self.mask, self.names, self.i = x, mask, names, 0

    def get_next(self):
        if self.i >= len(self.x):
            return None
        d = {self.names[0]: self.x[self.i:self.i + 1], self.names[1]: self.mask[self.i:self.i + 1]}
        self.i += 1
        return d


def quantise(fp32_path, int8_path, x, mask):
    pre = fp32_path.with_name(fp32_path.stem + "_pre.onnx")
    quant_pre_process(str(fp32_path), str(pre))
    names = [i.name for i in ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"]).get_inputs()]
    quantize_static(str(pre), str(int8_path), Calib(x, mask, names), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv", "Gemm", "MatMul"])
    pre.unlink()


def ort_vectors(path, table, byte_matrix):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    ids = byte_matrix.numpy().astype(np.int64)
    out = np.zeros((len(ids), 384), dtype=np.float32)
    for i in range(len(ids)):
        if ids[i].any():
            x, m = embed_inputs(table, ids[i:i + 1])
            out[i] = sess.run(None, {names[0]: x, names[1]: m})[0][0]
    return torch.tensor(np.nan_to_num(out))


train_news = data_mind.read_news(cfg, "train")
calib_ids = _byte_matrix([v["title"] for v in list(train_news.values())[:512]], L).astype(np.int64)

# ---------------------------------------------------------------- trained constrained encoder
sd = torch.load(CKPT, map_location="cpu")
enc = ByteCNNEncoder(byte_embed_dim=64, channels=64, depth=5, out_dim=384)
enc.load_state_dict({k[len("news_encoder."):]: v for k, v in sd.items() if k.startswith("news_encoder.")})
table = enc.embed.weight.detach().numpy().astype(np.float32)
files = {"micro_nas_64-5-384": (OUT / "micro_nas_64-5-384_body_fp32.onnx", OUT / "micro_nas_64-5-384_body_int8qdq.onnx")}
export(enc, files["micro_nas_64-5-384"][0])
cx, cm = embed_inputs(table, calib_ids)
quantise(*files["micro_nas_64-5-384"], cx, cm)

# ---------------------------------------------------------------- the two other searched architectures, random weights
torch.manual_seed(0)
for arch, C, D in (("nas_256-4-384", 256, 4), ("bin_unas_96-2-384", 96, 2)):
    e = ByteCNNEncoder(byte_embed_dim=64, channels=C, depth=D, out_dim=384)
    files[arch] = (OUT / f"{arch}_body_fp32.onnx", OUT / f"{arch}_body_int8qdq.onnx")
    export(e, files[arch][0])
    tx, tm = embed_inputs(e.embed.weight.detach().numpy().astype(np.float32), calib_ids)
    quantise(*files[arch], tx, tm)

# ---------------------------------------------------------------- accuracy of the deployed files (constrained encoder)
rec_torch = recommender.NewsRecommender(enc).to(device)
rec_torch.load_state_dict(sd)
rec_torch.eval()
vocab = recommender.eval_vocab(cfg, "dev")
results = {"checkpoint": str(CKPT),
           "input": "embedded sequence (1, 64, 128) and padding mask (1, 1, 128); the 257 x 64 byte table (65,792 B) is applied by the caller",
           "byte_table_bytes": int(table.size * 4), "calibration_titles": int(len(calib_ids)), "files": {}}
with torch.no_grad():
    results["torch_fp32"] = recommender.evaluate(cfg, rec_torch, "dev")
user_only = recommender.NewsRecommender(recommender.FixedVectors(384)).to(device)
user_only.load_state_dict({k: v for k, v in sd.items() if not k.startswith("news_encoder.")}, strict=False)
vec = {}
for key, path in (("onnx_fp32", files["micro_nas_64-5-384"][0]), ("onnx_int8qdq", files["micro_nas_64-5-384"][1])):
    t0 = time.time()
    V = ort_vectors(path, table, vocab.byte_matrix)
    vec[key] = V
    with torch.no_grad():
        results[key] = recommender.evaluate(cfg, user_only, "dev", news_vectors=V)
    results[key]["encode_minutes"] = round((time.time() - t0) / 60, 2)
    results["files"][key] = {"path": str(path), "bytes": path.stat().st_size}
    print(key, results[key])
cos = torch.nn.functional.cosine_similarity(vec["onnx_fp32"][1:], vec["onnx_int8qdq"][1:], dim=-1)
results["cosine_fp32_vs_int8"] = {"mean": float(cos.mean()), "min": float(cos.min()), "p05": float(cos.quantile(0.05))}
results["other_architectures"] = {k: {"fp32": str(v[0]), "int8qdq": str(v[1]), "weights": "random"} for k, v in files.items() if k != "micro_nas_64-5-384"}
results["minutes"] = round((time.time() - t_start) / 60, 2)
results["torch"] = torch.__version__
results["onnxruntime"] = ort.__version__
RES.write_text(json.dumps(results, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in results.items() if k not in ("files", "other_architectures")}, indent=1))
