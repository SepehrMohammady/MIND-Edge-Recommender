"""Integer files of the step-4 encoders and the ranking accuracy of each file, per seed.

Encoders, all click-trained from the distilled start on English clicks (the deployment
protocol) with seeds 42, 12 and 1:

  unas_h7       µNAS choice for the STM32H7B3I-DK budget   artifacts/runs/unas_full/mind_h7_distill_ft_en_seed*.pt
  unas_f401     µNAS choice for the NUCLEO-F401RE budget   artifacts/runs/unas_full/mind_f401_distill_ft_en_seed*.pt
  ref_64-5-384  hand-designed reference                    artifacts/runs/p1/distill_ft_en_seed*.pt
  ref_mixed_64-5-384  the same, clicks in a random language (the FeedWell-Edge encoder)
                                                           artifacts/runs/p1/distill_ft_mixed_seed*.pt
  hand_<C>-<D>-384  hand-designed family in the fork's terms (fixed table, mask channel, GAP):
                64-5-384 and the best of the step-3 grid within each budget, 64-2-384 (H7) and 32-5-384
                (F401)                                     artifacts/runs/unas_full/hand_*_distill_ft_en_seed*.pt

Export as in scripts/deploy_boards.py: the network after the byte-table lookup, inputs embedded
sequence (1, 64, 128) and padding mask (1, 1, 128), FP32 ONNX (opset 18), then ONNX Runtime
static quantisation to QDQ (per-channel symmetric int8 weights, int8 activations; the first 512
training titles for calibration), in two variants: "int8qdq" quantises Conv, Gemm and MatMul only
(the files measured on the boards on 2026-10-07), "int8full" every operator ONNX Runtime supports. Accuracy of each file on the 73,152 dev
impressions: news vectors from the file (ONNX Runtime, CPU), user encoder of the same checkpoint.
Checkpoints that do not exist yet are skipped; finished entries are kept on a rerun.

Writes artifacts/stedgeai/models/<name>_seed<seed>_body_{fp32,int8qdq,int8full}.onnx and
paper/results/int8_export.json. Runs on the CPU (the GPU belongs to the training queue):
    set CUDA_VISIBLE_DEVICES=-1 & python -m scripts.export_int8
"""
import json
import re
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from src import data_mind, recommender
from src.config import load_config
from src.student import ByteCNNEncoder, _byte_matrix
from src.unas_encoder import UnasEncoder, model_arch

cfg = load_config()
L = cfg["data"]["max_title_bytes"]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stedgeai/models"
OUT.mkdir(parents=True, exist_ok=True)
RES = Path(cfg["paths"]["results_dir"]) / "int8_export.json"
SEEDS = (42, 12, 1)
MODELS = {
    "unas_h7": ("mind_h7", ROOT / "artifacts/runs/unas_full/mind_h7_distill_ft_en_seed{seed}.pt"),
    "unas_f401": ("mind_f401", ROOT / "artifacts/runs/unas_full/mind_f401_distill_ft_en_seed{seed}.pt"),
    "ref_64-5-384": (None, ROOT / "artifacts/runs/p1/distill_ft_en_seed{seed}.pt"),
    "ref_mixed_64-5-384": (None, ROOT / "artifacts/runs/p1/distill_ft_mixed_seed{seed}.pt"),   # the app encoder (step 6)
}
for _arch in ("64-5-384", "64-2-384", "32-5-384"):
    MODELS[f"hand_{_arch}"] = (f"hand_{_arch}", ROOT / f"artifacts/runs/unas_full/hand_{_arch}_distill_ft_en_seed{{seed}}.pt")
# any µNAS candidate trained on English clicks under its candidate name (scripts/rechoose_unas.py: mind_h7_c140, ...)
for _key in json.loads((ROOT / "paper/results/unas_full.json").read_text(encoding="utf-8")):
    _m = re.fullmatch(r"(mind_[a-z0-9]+_c\d+)/distill_ft_en/seed\d+", _key)
    if _m:
        MODELS.setdefault(f"unas_{_m[1][len('mind_'):]}",
                          (_m[1], ROOT / f"artifacts/runs/unas_full/{_m[1]}_distill_ft_en_seed{{seed}}.pt"))
device = "cuda" if torch.cuda.is_available() else "cpu"


class Body(torch.nn.Module):
    """The encoder after the byte-table lookup: embedded sequence and mask in, news vector out."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, x, mask):                  # x (1, E, L) float32, mask (1, 1, L) float32
        if isinstance(self.enc, UnasEncoder):
            return self.enc.body(x, mask)
        x = self.enc.proj_in(x)
        for blk in self.enc.blocks:
            x = blk(x)
        x = (x * mask).sum(-1) / mask.sum(-1).clamp(min=1)
        return self.enc.head(x)


def embed_inputs(table, ids):
    """numpy lookup: ids (N, L) int -> x (N, E, L) float32, mask (N, 1, L) float32."""
    x = np.ascontiguousarray(table[ids].transpose(0, 2, 1)).astype(np.float32)
    return x, (ids != 0).astype(np.float32)[:, None, :]


def fold_pads(path):
    """Move every zero-valued constant Pad into the Conv that reads it (Conv "pads" attribute).

    The same function, without a Pad node: ST Edge AI Core 4.0.1 writes C code that does not
    compile for the Pad node of a PyTorch export (its constant value is left empty; network.c
    "expected expression before ']'", benchmark of 2026-10-08). The pads are computed by a small
    subgraph in the export, so the graph is first constant-folded with ONNX Runtime (basic level)."""
    folded = path.with_name(path.stem + "_folded.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(folded)
    ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    m = onnx.load(str(folded))
    folded.unlink()
    g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    users = {}
    for n in g.node:
        for i in n.input:
            users.setdefault(i, []).append(n)
    for pad in [n for n in g.node if n.op_type == "Pad"]:
        mode = next((a.s for a in pad.attribute if a.name == "mode"), b"constant")
        value = inits.get(pad.input[2]) if len(pad.input) > 2 and pad.input[2] else np.zeros(1)
        conv = users.get(pad.output[0], [])
        assert mode == b"constant" and pad.input[1] in inits and value is not None and not np.any(value), pad
        assert len(conv) == 1 and conv[0].op_type == "Conv", f"Pad {pad.name} does not feed one Conv"
        conv, p = conv[0], inits[pad.input[1]].tolist()
        r = len(p) // 2
        assert not any(p[:2]) and not any(p[r:r + 2]), "padding on batch or channel axis"
        old = next((list(a.ints) for a in conv.attribute if a.name == "pads"), [0] * (2 * (r - 2)))
        new = [old[i] + p[2 + i] for i in range(r - 2)] + [old[r - 2 + i] + p[r + 2 + i] for i in range(r - 2)]
        for a in [a for a in conv.attribute if a.name in ("pads", "auto_pad")]:
            conv.attribute.remove(a)
        conv.attribute.append(onnx.helper.make_attribute("pads", new))
        conv.input[0] = pad.input[0]
        g.node.remove(pad)
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def export(enc, path):
    # The exporter puts the wrapper back into the mode it had; a new wrapper is in training mode,
    # which would leave the encoder with BatchNorm on batch statistics and dropout active afterwards.
    enc.eval()
    torch.onnx.export(Body(enc).eval(), (torch.zeros(1, enc.embed.weight.shape[1], L), torch.ones(1, 1, L)), str(path),
                      input_names=["embedded_bytes", "mask"], output_names=["news_embedding"], opset_version=18,
                      dynamo=False)
    if any(n.op_type == "Pad" for n in onnx.load(str(path)).graph.node):
        fold_pads(path)
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


# Two integer files per encoder. "int8qdq": only Conv, Gemm and MatMul get quantize/dequantize pairs, so
# ReLU, pooling and the averages run in float between them (the files of 2026-10-07 and of the reference's
# board runs). "int8full": every operator ONNX Runtime can quantise (activations stay 8-bit through Concat,
# MaxPool, ReduceMean, Mul/Div; ReLU folds into the convolution's output range).
VARIANTS = {"onnx_int8qdq": ("int8qdq", ["Conv", "Gemm", "MatMul"]), "onnx_int8full": ("int8full", None)}


def quantise(fp32_path, int8_path, x, mask, ops=("Conv", "Gemm", "MatMul")):
    pre = fp32_path.with_name(fp32_path.stem + "_pre.onnx")
    quant_pre_process(str(fp32_path), str(pre))
    names = [i.name for i in ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"]).get_inputs()]
    quantize_static(str(pre), str(int8_path), Calib(x, mask, names), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=list(ops) if ops else None)
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


def build(search):
    if search is None:
        return ByteCNNEncoder(byte_embed_dim=64, channels=64, depth=5, out_dim=384)
    return UnasEncoder(model_arch(search)[1])


def metrics(res):
    return {k: round(v, 4) for k, v in res.items() if k != "n_impressions"} | {"n_impressions": res["n_impressions"]}


train_news = data_mind.read_news(cfg, "train")
calib_ids = _byte_matrix([v["title"] for v in list(train_news.values())[:512]], L).astype(np.int64)
vocab = recommender.eval_vocab(cfg, "dev")
done = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {"runs": {}}
for name, (search, pattern) in MODELS.items():
    for seed in SEEDS:
        ckpt, key = Path(str(pattern).format(seed=seed)), f"{name}/seed{seed}"
        r = done["runs"].get(key)
        missing = [v for v in VARIANTS if r is None or v not in r]
        if not missing or not ckpt.exists():
            continue
        t0 = time.time()
        sd = torch.load(ckpt, map_location="cpu")
        enc = build(search)
        enc.load_state_dict({k[len("news_encoder."):]: v for k, v in sd.items() if k.startswith("news_encoder.")})
        table = enc.embed.weight.detach().numpy().astype(np.float32)
        fp32 = OUT / f"{name}_seed{seed}_body_fp32.onnx"
        user_only = recommender.NewsRecommender(recommender.FixedVectors(384)).to(device)
        user_only.load_state_dict({k: v for k, v in sd.items() if not k.startswith("news_encoder.")}, strict=False)
        if r is None:
            export(enc, fp32)
            rec = recommender.NewsRecommender(enc).to(device)
            rec.load_state_dict(sd)
            rec.eval()
            r = {"checkpoint": str(ckpt.relative_to(ROOT)), "calibration_titles": int(len(calib_ids))}
            with torch.no_grad():
                r["torch_fp32"] = metrics(recommender.evaluate(cfg, rec, "dev"))
        vec = {"onnx_fp32": ort_vectors(fp32, table, vocab.byte_matrix)}
        if "onnx_fp32" not in r:
            with torch.no_grad():
                r["onnx_fp32"] = metrics(recommender.evaluate(cfg, user_only, "dev", news_vectors=vec["onnx_fp32"]))
            r["onnx_fp32"] |= {"file_bytes": fp32.stat().st_size, "file": str(fp32.relative_to(ROOT))}
        for kind in missing:
            suffix, ops = VARIANTS[kind]
            path = OUT / f"{name}_seed{seed}_body_{suffix}.onnx"
            quantise(fp32, path, *embed_inputs(table, calib_ids), ops=ops)
            vec[kind] = ort_vectors(path, table, vocab.byte_matrix)
            with torch.no_grad():
                r[kind] = metrics(recommender.evaluate(cfg, user_only, "dev", news_vectors=vec[kind]))
            r[kind] |= {"file_bytes": path.stat().st_size, "file": str(path.relative_to(ROOT))}
            cos = torch.nn.functional.cosine_similarity(vec["onnx_fp32"][1:], vec[kind][1:], dim=-1)
            r["cosine_fp32_vs_int8" if kind == "onnx_int8qdq" else f"cosine_fp32_vs_{suffix}"] = {
                "mean": round(float(cos.mean()), 4), "min": round(float(cos.min()), 4),
                "p05": round(float(cos.quantile(0.05)), 4)}
        r["minutes"] = round(r.get("minutes", 0) + (time.time() - t0) / 60, 2)
        done["runs"][key] = r
        RES.write_text(json.dumps(done, indent=1), encoding="utf-8")
        print(f"{key}: AUC torch {r['torch_fp32']['auc']:.4f}, ONNX FP32 {r['onnx_fp32']['auc']:.4f}, "
              + ", ".join(f"{VARIANTS[k][0]} {r[k]['auc']:.4f} ({r[k]['file_bytes']} B)" for k in VARIANTS)
              + f"; {r['minutes']} min", flush=True)

summary = {}
for name in MODELS:
    rows = [v for k, v in done["runs"].items() if k.startswith(name + "/")]
    if rows:
        summary[name] = {"n": len(rows)}
        for kind in ("torch_fp32", "onnx_fp32", *VARIANTS):
            vals = [r[kind]["auc"] for r in rows if kind in r]
            if vals:
                summary[name][f"{kind}_auc"] = round(statistics.fmean(vals), 4)
                if len(vals) > 1:
                    summary[name][f"{kind}_auc_sd"] = round(statistics.pstdev(vals), 4)
                if kind in VARIANTS:
                    summary[name][f"{VARIANTS[kind][0]}_minus_fp32_auc"] = round(statistics.fmean(
                        r[kind]["auc"] - r["torch_fp32"]["auc"] for r in rows if kind in r), 4)
done["summary"] = summary
done["versions"] = {"torch": torch.__version__, "onnxruntime": ort.__version__, "onnx": onnx.__version__}
RES.write_text(json.dumps(done, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
