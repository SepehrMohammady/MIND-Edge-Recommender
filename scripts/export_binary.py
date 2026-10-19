"""Export the one-bit ReActNet encoder (src/binary.py, scripts/run_binary.py) for Larq Compute Engine (schedule step 5).

    python -m scripts.export_binary                 # 1: weights, test inputs, PyTorch outputs, dev byte ids
    (WSL)  python scripts/lce_convert.py            # 2: Larq model, LCE TFLite file, LCE vectors (scripts/lce_convert.py)
    python -m scripts.export_binary --lce-vectors   # 3: AUC of the LCE file on MINDsmall dev

Step 1 writes artifacts/binary_lce/<name>_params.npz: the byte table, the full-precision first projection and
head, and per binary block the RSign threshold, the sign of the conv weights, and the per-channel affine that
follows the binary convolution, folded from the XNOR scale (mean |w| per output channel), the conv bias and the
batch norm (eval statistics):  out = mult * conv(sign(x - t), sign(w)) + add.  Then RPReLU (b1, a, b2) and the
real-valued shortcut. It also writes the inputs and PyTorch outputs of the 20 test titles of
scripts/board_validate.py and the byte ids of the dev vocabulary (recommender.eval_vocab), and the PyTorch AUC.
Step 3 reads the LCE vectors of step 2 and writes paper/results/binary_lce.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src import recommender
from src.binary import BinaryByteCNNEncoder
from src.config import load_config
from src.student import text_to_bytes

ROOT = Path(__file__).resolve().parents[1]
ARCH = dict(byte_embed_dim=64, channels=64, depth=5, out_dim=384)      # scripts/run_binary.py
OUT = ROOT / "artifacts/binary_lce"
cfg = load_config()


def metrics(res):
    return {k: round(v, 4) for k, v in res.items() if k != "n_impressions"} | {"n_impressions": res["n_impressions"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lce-vectors", action="store_true", help="step 3: AUC of the LCE file from its vectors")
    a = ap.parse_args()
    name = f"reactnet_64-5-384_seed{a.seed}"
    ckpt = ROOT / f"artifacts/runs/binary/{name}.pt"
    sd = torch.load(ckpt, map_location="cpu")
    enc = BinaryByteCNNEncoder(**ARCH)
    enc.load_state_dict({k[len("news_encoder."):]: v for k, v in sd.items() if k.startswith("news_encoder.")})
    enc.eval()
    user_only = recommender.NewsRecommender(recommender.FixedVectors(384))
    user_only.load_state_dict({k: v for k, v in sd.items() if not k.startswith("news_encoder.")}, strict=False)
    user_only.eval()
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = ROOT / "paper/results/binary_lce.json"
    t0 = time.time()

    if a.lce_vectors:
        lce = np.load(OUT / f"{name}_lce_dev_vectors.npy")
        conv = json.loads((OUT / f"{name}_lce_convert.json").read_text(encoding="utf-8"))
        res = json.loads(res_path.read_text(encoding="utf-8"))
        with torch.no_grad():
            res["lce_file"] = metrics(recommender.evaluate(cfg, user_only, "dev", news_vectors=torch.from_numpy(lce)))
        tv = np.load(OUT / f"{name}_torch_dev_vectors.npy")
        cos = (lce[1:] * tv[1:]).sum(1) / (np.linalg.norm(lce[1:], axis=1) * np.linalg.norm(tv[1:], axis=1) + 1e-12)
        res["lce_vs_torch_dev"] = {"mean_cosine": round(float(cos.mean()), 5), "min_cosine": round(float(cos.min()), 5),
                                   "share_below_0.999": round(float((cos < 0.999).mean()), 5), "titles": int(len(cos))}
        res["convert"] = conv
        res_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: res[k] for k in ("torch", "lce_file", "lce_vs_torch_dev")}, indent=1))
        return

    # weights, folded
    p = {"embed": enc.embed.weight.detach().numpy(), "proj_w": enc.proj_in.weight.detach().numpy()[:, :, 0],
         "proj_b": enc.proj_in.bias.detach().numpy(), "head_w": enc.head.weight.detach().numpy(),
         "head_b": enc.head.bias.detach().numpy()}
    for i, blk in enumerate(enc.blocks):
        w = blk.conv.weight.detach()
        scale = w.abs().mean(dim=(1, 2))
        bn = blk.bn
        g = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
        p[f"b{i}_thr"] = blk.sign.bias.detach().numpy().reshape(-1)
        p[f"b{i}_wsign"] = torch.where(w >= 0, 1.0, -1.0).numpy()                       # (out, in, k)
        p[f"b{i}_mult"] = (g * scale).numpy()
        p[f"b{i}_add"] = (g * (blk.conv.bias.detach() - bn.running_mean) + bn.bias.detach()).numpy()
        p[f"b{i}_rp_b1"] = blk.act.b1.detach().numpy().reshape(-1)
        p[f"b{i}_rp_a"] = blk.act.a.detach().numpy().reshape(-1)
        p[f"b{i}_rp_b2"] = blk.act.b2.detach().numpy().reshape(-1)
    np.savez(OUT / f"{name}_params.npz", **{k: v.astype(np.float32) for k, v in p.items()})

    # the 20 test titles of scripts/board_validate.py, and the dev vocabulary
    from scripts.board_validate import titles
    rows = titles()
    L = cfg["data"]["max_title_bytes"]
    ids = np.array([text_to_bytes(t, L) for _, _, t in rows], dtype=np.int64)
    vocab = recommender.eval_vocab(cfg, "dev")
    dev_ids = vocab.byte_matrix.numpy().astype(np.int64)
    with torch.no_grad():
        ref = enc(torch.from_numpy(ids)).numpy()
        tv = np.concatenate([enc(torch.from_numpy(dev_ids[i:i + 512])).numpy() for i in range(0, len(dev_ids), 512)])
    tv[~dev_ids.any(1)] = 0.0                       # empty rows (padding) as in the other exports
    np.savez(OUT / f"{name}_test.npz", ids=ids, torch_out=ref, langs=np.array([r[0] for r in rows]),
             nids=np.array([r[1] for r in rows]))
    np.save(OUT / f"{name}_dev_ids.npy", dev_ids)
    np.save(OUT / f"{name}_torch_dev_vectors.npy", tv.astype(np.float32))
    with torch.no_grad():
        auc_torch = metrics(recommender.evaluate(cfg, user_only, "dev", news_vectors=torch.from_numpy(tv)))
    res = {"model": name, "checkpoint": str(ckpt.relative_to(ROOT)), "arch": ARCH, "torch": auc_torch,
           "test_titles": [{"lang": lg, "nid": n} for lg, n, _ in rows], "dev_titles": int(len(dev_ids)),
           "minutes_export": round((time.time() - t0) / 60, 2)}
    res_path.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("model", "torch", "dev_titles")}, indent=1))


if __name__ == "__main__":
    main()
