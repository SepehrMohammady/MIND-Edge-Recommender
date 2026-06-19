"""Tiny sanity check for the NAS search and the NRMS baseline."""
from src.config import load_config
from src.nas import search
from src import baseline_nrms

cfg = load_config()

# --- NAS (micro_nas / INT8, tiny budget) ---
fit = search.make_distill_fitness(cfg, "int8", n_train=3000, n_val=1000, epochs=1, batch=256)
res = search.search(cfg, "micro_nas", generations=2, population=6, fitness_fn=fit)
print("top micro_nas:", {k: res[0][k] for k in ("arch", "quality", "size_kb", "ram_kb", "macs", "feasible")})

# --- NRMS baseline ---
m = baseline_nrms.train_nrms(cfg, epochs=1, max_train_impressions=400)
print("NRMS dev:", baseline_nrms.evaluate_nrms(cfg, m, split="dev", max_impressions=300))
print("NAS+NRMS SMOKE OK")
