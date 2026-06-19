"""Fast end-to-end sanity check (no teacher needed): random-init student ->
train a few hundred impressions -> evaluate on a few hundred. Catches shape/
logic bugs before the full runs. Run:  python -m scripts.smoke_test
"""
import torch

from src.config import load_config
from src import recommender, footprint
from src.student import ByteCNNEncoder

cfg = load_config()
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| cap", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)

cfg["train"]["batch_size"] = 32
model = recommender.train_recommender(cfg, epochs=1, max_train_impressions=400)

res_en = recommender.evaluate(cfg, model, split="dev", max_impressions=300)
print("EN dev (untrained-ish):", res_en)

res_ro = recommender.evaluate(cfg, model, split="dev", lang="ron", max_impressions=300)
print("RON dev (cross-lingual):", res_ro)

enc = ByteCNNEncoder(cfg["student"]["byte_embed_dim"], cfg["student"]["channels"],
                     cfg["student"]["depth"], cfg["student"]["out_dim"])
ex = torch.zeros(1, cfg["data"]["max_title_bytes"], dtype=torch.long)
print("encoder footprint fp32:", footprint.summarize(enc, ex, "fp32", cfg))
print("SMOKE OK")
