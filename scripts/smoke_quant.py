"""Quick check that INT8/Binary conversion + eval + footprint all run."""
import torch

from src.config import load_config
from src import recommender, footprint, quantize

cfg = load_config()
cfg["train"]["batch_size"] = 32
model = recommender.train_recommender(cfg, epochs=1, max_train_impressions=400)

ex = torch.zeros(1, cfg["data"]["max_title_bytes"], dtype=torch.long)
for prec in ["fp32", "int8", "binary"]:
    qm = quantize.convert_to_quant(model, prec).to(next(model.parameters()).device)
    res = recommender.evaluate(cfg, qm, split="dev", max_impressions=200)
    fp_frac = quantize.quant_fp_fraction(qm.news_encoder)
    foot = footprint.summarize(qm.news_encoder, ex, prec, cfg, fp_fraction=fp_frac)
    print(f"{prec:7s} AUC={res['auc']:.4f}  fp_frac={fp_frac:.2f}  "
          f"size_kb={foot['size_kb']}  energy_uJ={foot['energy_uj_per_inf']}")
print("QUANT SMOKE OK")
