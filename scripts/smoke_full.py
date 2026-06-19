"""End-to-end orchestration smoke: train -> precision sweep -> multilingual ->
ONNX export -> latency/energy. Tiny settings."""
import torch

from src.config import load_config
from src import experiment, export, measure_energy

cfg = load_config()
arch = {"channels": 64, "depth": 2, "out_dim": 128}

m = experiment.train_for_arch(cfg, arch, epochs=1, max_train_impressions=400)

print("--- precision sweep ---")
for r in experiment.precision_sweep(cfg, m, arch, "micro_nas", qat_epochs=0, max_eval=200):
    print(r)

print("--- multilingual ---")
print(experiment.eval_languages(cfg, m, langs=["en", "ron", "jpn"], max_impressions=200))

onnx = export.export_encoder_onnx(m.news_encoder.cpu(), cfg)
print("onnx ->", onnx)
print("onnx CPU latency ms:", round(measure_energy.latency_onnx(onnx, cfg["data"]["max_title_bytes"], n=50), 4))

m = m.to("cuda" if torch.cuda.is_available() else "cpu")
ex = torch.zeros(8, cfg["data"]["max_title_bytes"], dtype=torch.long)
print("gpu latency ms:", round(measure_energy.latency_torch(m.news_encoder, ex), 4))
print("energy:", measure_energy.energy_nvml(m.news_encoder, ex, n=200))

state = export.export_artifacts(cfg, m.news_encoder.cpu(), ["en", "ron"])
print("state ->", state)
print("FULL SMOKE OK")
