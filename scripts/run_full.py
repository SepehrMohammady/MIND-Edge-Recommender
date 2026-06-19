"""Full paper-grade run on MINDsmall: distill -> NAS (3 arms) -> train ->
precision sweep (with QAT) -> NRMS baseline -> all-14-language eval -> export.
Saves every table to artifacts/. Run: python -m scripts.run_full
"""
import json
import time
from pathlib import Path

import pandas as pd

from src.config import load_config
from src import data_xmind, experiment, export

cfg = load_config()
cfg["train"]["distill_epochs"] = 15
cfg["train"]["epochs"] = 8
art = Path(cfg["paths"]["artifacts_dir"])
t0 = time.time()


def log(msg):
    print(f"[{(time.time()-t0)/60:6.1f}m] {msg}", flush=True)


log("anchors")
experiment.ensure_anchors(cfg)

log("distill student")
experiment.distill(cfg)

log("NAS (3 arms)")
arms = experiment.search_arms(cfg, generations=15, population=30,
                              n_train=20000, n_val=4000, distill_epochs=2)
best = {a: experiment.best_arch(r) for a, r in arms.items()}
log(f"best archs: {best}")

log("train recommender per arm")
models = {a: experiment.train_for_arch(cfg, best[a]) for a in best}

log("precision sweep (QAT)")
rows = []
for a in models:
    rows += experiment.precision_sweep(cfg, models[a], best[a], a, qat_epochs=2, max_eval=None)
matrix = experiment.save_matrix(cfg, rows)
log("results matrix saved")

log("NRMS baseline")
baseline = experiment.baseline_row(cfg, epochs=8, max_eval=None)

log("all-14-language eval")
deploy = models["micro_nas"]
lang = experiment.eval_languages(cfg, deploy, max_impressions=None)
lang.to_csv(art / "lang_matrix.csv", index=False)

log("export artifacts")
state = export.export_artifacts(cfg, deploy.news_encoder.cpu(),
                                ["en"] + data_xmind.available_langs(cfg))

summary = {"best_arch": best, "baseline": baseline,
           "matrix": matrix.to_dict("records"),
           "languages": lang.to_dict("records"),
           "runtime_min": round((time.time() - t0) / 60, 1)}
(art / "results_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
log(f"FULL RUN DONE -> {art/'results_summary.json'}")
print(matrix.to_string())
print(lang.to_string())
print("baseline:", baseline)
