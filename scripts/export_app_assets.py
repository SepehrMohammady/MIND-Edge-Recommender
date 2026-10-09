"""Model files for the FeedWell-Edge app (schedule step 6), from one trained recommender.

Encoder: 64-5-384, distilled start, clicks shown in a random one of 15 languages (scripts/run_p1.py,
stage student, distill_ft_mixed, seed 42): the best multilingual encoder of Table 2 at this size.
Written to artifacts/app/edge_encoder_v1/:

  news_encoder_int8.onnx   8-bit QDQ file of scripts/export_int8.py (inputs: embedded bytes (1, 64, 128)
  news_encoder_fp32.onnx   and padding mask (1, 1, 128); output: 384-d news vector)
  byte_table.f32           the 257 x 64 byte table, float32 little-endian (the caller's lookup)
  user_encoder.onnx        additive attention over the history: vectors (1, H, 384) and mask (1, H) in,
                           user vector (1, 384) out (score = dot product with a news vector)
  user_encoder.json        the same weights as plain arrays, for a JavaScript implementation
  test_vectors.json        titles in several scripts with their byte ids, the news vectors of PyTorch and of
                           both ONNX files, and one history with candidates and their scores, to check the
                           phone's results against
  manifest.json            sources, SHA-256 and sizes of the files, and the accuracy of the 8-bit file

Accuracy of the 8-bit file: dev AUC in English and in each of the 14 xMIND languages (news vectors from
the file, user encoder of the checkpoint); also written to paper/results/app_encoder.json.
Byte ids follow src/student.text_to_bytes: UTF-8 bytes of the title, first 128, id = byte + 1, 0 = padding.

    python -m scripts.export_app_assets        (CPU)
"""
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src import data_xmind, recommender
from src.config import load_config
from src.student import ByteCNNEncoder, text_to_bytes

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "artifacts/runs/p1/distill_ft_mixed_seed42.pt"
SRC = ROOT / "artifacts/stedgeai/models"
OUT = ROOT / "artifacts/app/edge_encoder_v1"
OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()
L = cfg["data"]["max_title_bytes"]
t0 = time.time()

sd = torch.load(CKPT, map_location="cpu")
enc = ByteCNNEncoder(byte_embed_dim=64, channels=64, depth=5, out_dim=384)
enc.load_state_dict({k[len("news_encoder."):]: v for k, v in sd.items() if k.startswith("news_encoder.")})
enc.eval()
rec = recommender.NewsRecommender(enc)
rec.load_state_dict(sd)
rec.eval()
table = enc.embed.weight.detach().numpy().astype("<f4")

# ---- files
for kind in ("int8qdq", "fp32"):
    src = SRC / f"ref_mixed_64-5-384_seed42_body_{kind}.onnx"
    dst = OUT / f"news_encoder_{'int8' if kind == 'int8qdq' else 'fp32'}.onnx"
    dst.write_bytes(src.read_bytes())
(OUT / "byte_table.f32").write_bytes(table.tobytes())


class UserEncoder(torch.nn.Module):
    def __init__(self, r):
        super().__init__()
        self.r = r

    def forward(self, hist, mask):                # (1, H, 384), (1, H) -> (1, 384)
        return self.r.user_vector(hist, mask)


torch.onnx.export(UserEncoder(rec).eval(), (torch.zeros(1, 5, 384), torch.ones(1, 5)), str(OUT / "user_encoder.onnx"),
                  input_names=["history_vectors", "history_mask"], output_names=["user_vector"],
                  dynamic_axes={"history_vectors": {1: "H"}, "history_mask": {1: "H"}}, opset_version=18, dynamo=False)
onnx.checker.check_model(onnx.load(str(OUT / "user_encoder.onnx")))
(OUT / "user_encoder.json").write_text(json.dumps({
    "formula": "a_i = v . tanh(W h_i + b), masked softmax over i (mask 0 -> -1e4), user = sum_i softmax(a)_i h_i",
    "W": rec.attn.weight.detach().numpy().round(7).tolist(), "b": rec.attn.bias.detach().numpy().round(7).tolist(),
    "v": rec.attn_v.weight.detach().numpy()[0].round(7).tolist()}), encoding="utf-8")


def embed_inputs(ids):
    x = np.ascontiguousarray(table[ids].transpose(0, 2, 1)).astype(np.float32)
    return x, (ids != 0).astype(np.float32)[:, None, :]


sess = {k: ort.InferenceSession(str(OUT / f"news_encoder_{k}.onnx"), providers=["CPUExecutionProvider"])
        for k in ("int8", "fp32")}


def ort_vec(kind, ids):
    names = [i.name for i in sess[kind].get_inputs()]
    out = []
    for row in ids:
        if not row.any():
            out.append(np.zeros(384, np.float32))
            continue
        x, m = embed_inputs(row[None])
        out.append(sess[kind].run(None, {names[0]: x, names[1]: m})[0][0])
    return np.stack(out)


# ---- test vectors: titles in several scripts (Latin, Cyrillic-free Romanian, Chinese, Japanese, Georgian, Persian,
# Italian), a history and candidates
titles = ["Stocks rally as inflation cools", "Hurricane season forecast raised",
          "NASA delays moon mission after engine test", "Bursele cresc pe fondul scaderii inflatiei",
          "通胀降温，股市上涨", "インフレ鈍化で株価上昇", "ინფლაციის შემცირებით აქციები იზრდება",
          "بازار سهام با کاهش تورم رشد کرد", "Le borse salgono mentre l'inflazione rallenta",
          "Rugby: Italia batte la Francia a Roma", "New smartphone launch draws long lines", ""]
ids = np.array([text_to_bytes(t, L) for t in titles], dtype=np.int64)
with torch.no_grad():
    v_torch = enc(torch.from_numpy(ids)).numpy() * (ids != 0).any(-1, keepdims=True)
v_fp32, v_int8 = ort_vec("fp32", ids), ort_vec("int8", ids)
hist_idx, cand_idx = [0, 2, 10], [1, 3, 8, 9]
with torch.no_grad():
    scores = {}
    for name, V in (("torch", v_torch), ("onnx_fp32", v_fp32), ("onnx_int8", v_int8)):
        Vt = torch.from_numpy(V)
        u = rec.user_vector(Vt[hist_idx][None], torch.ones(1, len(hist_idx)))[0]
        scores[name] = (Vt[cand_idx] @ u).numpy().round(6).tolist()
(OUT / "test_vectors.json").write_text(json.dumps({
    "titles": titles, "byte_ids": ids.tolist(),
    "news_vectors": {"torch": v_torch.round(6).tolist(), "onnx_fp32": v_fp32.round(6).tolist(),
                     "onnx_int8": v_int8.round(6).tolist()},
    "history": hist_idx, "candidates": cand_idx, "candidate_scores": scores}, ensure_ascii=False), encoding="utf-8")
cos = lambda a, b: float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))  # noqa: E731
agree = {"torch_vs_fp32_max_abs": float(np.abs(v_torch - v_fp32).max()),
         "fp32_vs_int8_cos_min": min(cos(a, b) for a, b in zip(v_fp32[:-1], v_int8[:-1]))}

# ---- accuracy of the 8-bit file, English and the 14 xMIND languages (full dev set)
user_only = recommender.NewsRecommender(recommender.FixedVectors(384))
user_only.load_state_dict({k: v for k, v in sd.items() if not k.startswith("news_encoder.")}, strict=False)
user_only.eval()
acc = {}
for lang in ["en"] + data_xmind.available_langs(cfg):
    vocab = recommender.eval_vocab(cfg, "dev", None if lang == "en" else lang)
    V = torch.tensor(np.nan_to_num(ort_vec("int8", vocab.byte_matrix.numpy().astype(np.int64))))
    with torch.no_grad():
        r = recommender.evaluate(cfg, user_only, "dev", lang=None if lang == "en" else lang, news_vectors=V)
    acc[lang] = round(r["auc"], 4)
    print(f"{lang}: 8-bit file AUC {acc[lang]:.4f}", flush=True)
x = [v for k, v in acc.items() if k != "en"]
p1 = json.loads((ROOT / "paper/results/p1_student.json").read_text(encoding="utf-8"))["distill_ft_mixed/seed42"]["results"]
result = {"checkpoint": str(CKPT.relative_to(ROOT)), "encoder": "64-5-384, distilled start, mixed-language clicks, seed 42",
          "int8_file_auc": acc, "int8_file_mean_xlang_auc": round(statistics.fmean(x), 4),
          "torch_auc": {"en": p1["en"]["all"]["auc"], "mean_xlang": p1["mean_xlang_auc"]},
          "agreement_on_test_titles": agree, "minutes": round((time.time() - t0) / 60, 2)}
files = {}
for p in sorted(OUT.iterdir()):
    if p.name != "manifest.json":
        files[p.name] = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
(OUT / "manifest.json").write_text(json.dumps({**result, "files": files, "byte_ids": "UTF-8 bytes, first 128, id = byte + 1, 0 = padding",
                                               "versions": {"torch": torch.__version__, "onnxruntime": ort.__version__}},
                                              indent=1, ensure_ascii=False), encoding="utf-8")
(ROOT / "paper/results/app_encoder.json").write_text(json.dumps({**result, "files": files}, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k != "int8_file_auc"}, indent=1))
