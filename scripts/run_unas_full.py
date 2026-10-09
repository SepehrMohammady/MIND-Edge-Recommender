"""Full training of the encoders chosen by the µNAS search (step 4).

The search in the lab fork (unas/) chose one architecture per board from validation cosine
over three seeds: mind_h7 candidate 144 (STM32H7B3I-DK budget) and mind_f401 candidate 134
(NUCLEO-F401RE budget). During the search each candidate was distilled for at most 15 epochs
on 96,000 rows. Here the two chosen ones go through the programme of the reference encoder
64-5-384 (scripts/run_p1.py, stage student), so that their results compare run for run:

  distillation      15 epochs on every training title in 15 languages (769k rows), Matryoshka
                    cosine loss to the teacher's English anchor (src/student.py)
  distill_ft_en     click training from the distilled encoder, English clicks, 8 epochs
  distill_ft_mixed  the same, each click shown in a random one of the 15 languages

Seeds 42, 12 and 1, as in the reference runs (data order and negatives then match run for
run). The hand-designed family is trained the same way in the fork's terms ("hand_<C>-<D>-384":
fixed byte table, mask channel, global average pooling; src/unas_encoder.hand_designed), so that
the searched encoders meet hand-designed ones of the same budget, and the reference 64-5-384, with
the same input and pooling: hand_64-2-384 is the best of the step-3 grid within the H7 budget,
hand_32-5-384 the best within the F401 budget. Evaluation on the full dev set: English by
history bucket and the 14 xMIND languages. The network is the PyTorch build of src/unas_encoder.py, checked against the fork's Keras build
(scripts/check_unas_port.py); the byte table stays fixed, as in the search.

Results: paper/results/unas_full.json and paper/results/experiments.jsonl, one record per run;
checkpoints in artifacts/runs/unas_full. Finished runs are skipped on restart.

Run:  python -m scripts.run_unas_full [--models mind_h7 mind_f401] [--protocols distill_ft_en]
                                     [--seeds 42 12 1] [--smoke]
"""
import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch

from src import data_xmind, footprint, recommender, runlog, student
from src.config import load_config, use_run_dir
from src.gpu import cap_gpu_memory
from src.seed import seed_everything
from src.unas_encoder import UnasEncoder, model_arch

PROTOCOLS = {"distill_ft_en": ["en"], "distill_ft_mixed": None}      # None = all 15 languages
EPOCHS, DISTILL_EPOCHS = 8, 15
TRAIN_IMPRESSIONS = EVAL_IMPRESSIONS = DISTILL_TITLES = None          # None = everything

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--models", nargs="+", default=["mind_h7", "mind_f401"],
                    help="mind_h7, mind_f401 (the search's choices), mind_h7_c<index> (any candidate) "
                         "or hand_<C>-<D>-384")
parser.add_argument("--protocols", nargs="+", default=["distill_ft_en"], choices=list(PROTOCOLS) + ["distill"],
                    help='"distill" alone: distillation and its record only (scripts/rechoose_unas.py)')
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 12, 1])
parser.add_argument("--smoke", action="store_true", help="tiny settings, two languages, results kept out of paper/results")
args = parser.parse_args()

cfg = load_config()
use_run_dir(cfg, "unas_full_smoke" if args.smoke else "unas_full")
cap_gpu_memory()                    # src/gpu.py: keeps the allocator's cache inside the card
CKPT = Path(cfg["paths"]["artifacts_dir"])
XLANGS = data_xmind.available_langs(cfg)
if args.smoke:
    EPOCHS, DISTILL_EPOCHS = 1, 1
    TRAIN_IMPRESSIONS, EVAL_IMPRESSIONS, DISTILL_TITLES = 2000, 1000, 20000
    XLANGS = XLANGS[:2]
    cfg["paths"]["results_dir"] = str(CKPT / "results")
    Path(cfg["paths"]["results_dir"]).mkdir(parents=True, exist_ok=True)
RESULTS = Path(cfg["paths"]["results_dir"])
OUT = RESULTS / "unas_full.json"
LANGS = ["en"] + XLANGS
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{(time.time() - T0) / 60:7.1f}m] {msg}", flush=True)


def load_done() -> dict:
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}


def save_run(key: str, settings: dict, results: dict, started: float) -> None:
    record = runlog.append(cfg, f"unas_full/{key}", settings, results, started)
    done = load_done()
    done[key] = {k: record[k] for k in ("seed", "minutes", "settings", "results")}
    OUT.write_text(json.dumps(done, indent=1), encoding="utf-8")
    msg = f"{key}: {record['minutes']} min"
    if "en" in results:
        msg += f", EN AUC {results['en']['all']['auc']:.4f}, mean of 14 languages {results['mean_xlang_auc']:.4f}"
    log(msg)


def rounded(d: dict) -> dict:
    return {k: (rounded(v) if isinstance(v, dict) else round(v, 4) if isinstance(v, float) else v)
            for k, v in d.items()}


def language_results(model) -> dict:
    """English by history bucket, then every xMIND language, full dev set (as scripts/run_p1.py)."""
    out = {"en": rounded(recommender.evaluate_by_history(cfg, model, max_impressions=EVAL_IMPRESSIONS))}
    for lang in XLANGS:
        out[lang] = rounded(recommender.evaluate(cfg, model, lang=lang, max_impressions=EVAL_IMPRESSIONS))
    out["mean_xlang_auc"] = round(statistics.fmean(out[l]["auc"] for l in XLANGS), 4)
    return out


def distilled_encoder(name: str, label, arch: dict, seed: int, cost: dict) -> UnasEncoder:
    path = CKPT / f"{name}_distilled_seed{seed}.pt"
    seed_everything(seed)
    enc = UnasEncoder(arch)
    if path.exists():
        enc.load_state_dict(torch.load(path, map_location="cpu"))
        return enc
    started = time.time()
    enc = student.distill_encoder(cfg, enc, epochs=DISTILL_EPOCHS, max_titles=DISTILL_TITLES, tag=f"distill {name}").cpu()
    torch.save(enc.state_dict(), path)
    teacher_cos = student.teacher_similarity(cfg, copy.deepcopy(enc).to(recommender_device()))
    save_run(f"{name}/distill/seed{seed}",
             {"model": label, "arch": arch, "distill_epochs": DISTILL_EPOCHS, "distill_titles": DISTILL_TITLES,
              "loss": "Matryoshka cosine (64, 128, 256, 384)", "byte_table": "fixed", "encoder_cost": cost},
             {"teacher_cosine_dev": teacher_cos}, started)
    return enc


def recommender_device() -> str:
    return "cuda" if torch.cuda.is_available() and cfg["train"]["device"] == "cuda" else "cpu"


def run_model(name: str, seed: int) -> None:
    label, arch = model_arch(name)
    todo = [p for p in args.protocols if p != "distill" and f"{name}/{p}/seed{seed}" not in load_done()]
    if not todo and f"{name}/distill/seed{seed}" in load_done():
        return
    cfg["seed"] = seed
    ex = torch.zeros(1, cfg["data"]["max_title_bytes"], dtype=torch.long)
    cost = footprint.summarize(UnasEncoder(arch), ex, cfg, precision="fp32")
    distilled = distilled_encoder(name, label, arch, seed, cost)
    for protocol in todo:
        started = time.time()
        seed_everything(seed)
        langs = PROTOCOLS[protocol] or LANGS
        model = recommender.train_recommender(cfg, news_encoder=copy.deepcopy(distilled), epochs=EPOCHS,
                                              max_train_impressions=TRAIN_IMPRESSIONS, langs=langs)
        torch.save(model.state_dict(), CKPT / f"{name}_{protocol}_seed{seed}.pt")
        settings = {"model": label, "arch": arch, "init": "distilled", "train_langs": langs, "epochs": EPOCHS,
                    "distill_epochs": DISTILL_EPOCHS, "byte_table": "fixed", "encoder_cost": cost}
        save_run(f"{name}/{protocol}/seed{seed}", settings, language_results(model), started)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    log(f"models={args.models} protocols={args.protocols} seeds={args.seeds} langs={len(LANGS)} smoke={args.smoke} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    for seed in args.seeds:
        for name in args.models:
            run_model(name, seed)
    log("UNAS FULL DONE")
