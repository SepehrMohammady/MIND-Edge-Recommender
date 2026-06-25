"""Improved-binary experiment: distil a ReActNet/Bi-Real byte-CNN to the teacher
anchors, fine-tune the recommender, evaluate on MINDsmall-dev. Compares against
the naive-binary numbers (Micro-NAS/Binary 0.521, Bin.uNAS/Binary 0.546).
Run: python -m scripts.run_binary
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import load_config
from src import student, recommender
from src.binary import BinaryByteCNNEncoder

cfg = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
ARCH = dict(byte_embed_dim=64, channels=64, depth=5, out_dim=384)   # Micro-NAS shape
DISTILL_EP, TRAIN_EP = 15, 8

bytes_np, tgt_np, anchors = student.build_distill_data(cfg, "train")
anchors_t = torch.tensor(anchors, device=device)
dl = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(bytes_np, dtype=torch.long),
                                   torch.tensor(tgt_np, dtype=torch.long)),
    batch_size=512, shuffle=True, drop_last=True)

enc = BinaryByteCNNEncoder(**ARCH).to(device)
opt = torch.optim.AdamW(enc.parameters(), lr=cfg["train"]["distill_lr"])
dims = (64, 128, 256, 384)
enc.train()
for ep in range(DISTILL_EP):
    last = 0.0
    for ids, ti in dl:
        ids = ids.to(device); tgt = anchors_t[ti.to(device)]
        pred = F.normalize(enc(ids), dim=-1)
        loss = sum(1 - F.cosine_similarity(pred[:, :d], tgt[:, :d], dim=-1).mean()
                   for d in dims) / len(dims)
        opt.zero_grad(); loss.backward(); opt.step()
        last = loss.item()
    print(f"[bin-distill] ep {ep+1}/{DISTILL_EP} loss={last:.4f}", flush=True)

model = recommender.train_recommender(cfg, model=recommender.NewsRecommender(enc), epochs=TRAIN_EP)
res = recommender.evaluate(cfg, model, split="dev")
print("IMPROVED BINARY (distilled-init, ReActNet/Bi-Real):", res)
Path(cfg["paths"]["artifacts_dir"], "binary_improved.json").write_text(
    json.dumps({"arch": ARCH, "result": {k: round(v, 4) for k, v in res.items()}}, indent=2))
