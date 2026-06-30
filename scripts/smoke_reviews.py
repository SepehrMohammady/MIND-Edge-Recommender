"""Fast sanity for run_reviews building blocks."""
import pathlib
import torch

from src.config import load_config
from src import recommender, quantize
from src.nas.search import build_encoder
from src.binary import BinaryByteCNNEncoder
from src.student import ByteCNNEncoder

cfg = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
arch = {"channels": 32, "depth": 2, "out_dim": 128}

m = recommender.train_recommender(cfg, news_encoder=build_encoder(cfg, arch), epochs=1, max_train_impressions=300)
qm = quantize.convert_to_quant(m, "int8").to(device)
print("int8 auc:", round(recommender.evaluate(cfg, qm, split="dev", max_impressions=200)["auc"], 4))
print("warm:", round(recommender.evaluate(cfg, m, split="dev", max_impressions=200, mask_history=False)["auc"], 4))
print("cold(mask):", round(recommender.evaluate(cfg, m, split="dev", max_impressions=200, mask_history=True)["auc"], 4))

enc = BinaryByteCNNEncoder(64, 32, 2, 128).to(device)
print("binary fwd:", tuple(enc(torch.randint(0, 257, (4, 128), device=device)).shape))

s = cfg["student"]
e = ByteCNNEncoder(s["byte_embed_dim"], s["channels"], s["depth"], s["out_dim"])
sp = pathlib.Path(cfg["paths"]["artifacts_dir"]) / "student.pt"
e.load_state_dict(torch.load(sp, map_location="cpu"))
print("student.pt loaded:", sp.exists())
print("SMOKE OK")
