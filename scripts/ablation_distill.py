"""Ablation: does teacher distillation help the deployed recommender?

Controlled comparison at the Micro-NAS architecture (64-5-384), MINDsmall-dev:
  * scratch        : byte-CNN news encoder trained end-to-end on clicks only
  * distilled-init : same encoder pre-distilled to the teacher anchors, then
                     fine-tuned end-to-end on clicks
Everything else (arch, epochs, data, seed) identical. Saves artifacts/ablation.json.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import load_config
from src import recommender, student
from src.student import ByteCNNEncoder

cfg = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
ARCH = dict(byte_embed_dim=64, channels=64, depth=5, out_dim=384)
EPOCHS = cfg["train"]["epochs"]          # match the main run (8)


def distill_encoder(epochs=12, dims=(64, 128, 256, 384)):
    bytes_np, tgt_np, anchors = student.build_distill_data(cfg, "train")
    anchors_t = torch.tensor(anchors, device=device)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(bytes_np, dtype=torch.long),
                                       torch.tensor(tgt_np, dtype=torch.long)),
        batch_size=512, shuffle=True, drop_last=True)
    enc = ByteCNNEncoder(**ARCH).to(device)
    opt = torch.optim.AdamW(enc.parameters(), lr=cfg["train"]["distill_lr"])
    enc.train()
    for ep in range(epochs):
        for ids, ti in dl:
            ids = ids.to(device); tgt = anchors_t[ti.to(device)]
            pred = F.normalize(enc(ids), dim=-1)
            loss = sum(1 - F.cosine_similarity(pred[:, :d], tgt[:, :d], dim=-1).mean()
                       for d in dims) / len(dims)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[ablation/distill] epoch {ep+1}/{epochs} loss={loss.item():.4f}", flush=True)
    return enc


def run():
    print("=== scratch (no distillation) ===", flush=True)
    scratch = recommender.train_recommender(cfg, news_encoder=ByteCNNEncoder(**ARCH), epochs=EPOCHS)
    r_scratch = recommender.evaluate(cfg, scratch, split="dev")

    print("=== distilled-init ===", flush=True)
    enc = distill_encoder()
    model = recommender.NewsRecommender(enc)
    distilled = recommender.train_recommender(cfg, model=model, epochs=EPOCHS)
    r_distilled = recommender.evaluate(cfg, distilled, split="dev")

    out = {"arch": ARCH, "epochs": EPOCHS,
           "scratch": {k: round(v, 4) for k, v in r_scratch.items()},
           "distilled_init": {k: round(v, 4) for k, v in r_distilled.items()}}
    Path(cfg["paths"]["artifacts_dir"], "ablation.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("ABLATION DONE:", json.dumps(out, indent=2))


if __name__ == "__main__":
    run()
