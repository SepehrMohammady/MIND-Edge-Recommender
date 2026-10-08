"""Design experiments on the reference byte-CNN (64-5-384), MINDsmall + xMIND.

Each stage answers one question the June 2026 run left open, with seeded,
repeated runs. Results are written after every run to
``paper/results/p1_<stage>.json`` and ``paper/results/experiments.jsonl``;
finished runs are skipped on restart, checkpoints go to ``artifacts/runs/p1``.

  heuristics  How far do popularity and a category histogram get, with no
              network at all?
  teacher     What does the frozen multilingual teacher reach as news encoder
              (upper bound of distillation), in English and per language?
  student     Scratch vs distilled initialisation; English clicks vs clicks
              shown in a random language; evaluated in all 15 languages.
  nrms        NRMS with the reference GloVe configuration and the reduced June
              variant, under the same training loop.

Run:  python -m scripts.run_p1 [stage ...] [--seeds 42 1 2]
"""
import argparse
import copy
import json
import statistics
import time
import traceback
from pathlib import Path

import torch

from src import (baseline_nrms, data_xmind, footprint, heuristics, recommender, runlog,
                 student, teacher)
from src.config import load_config, use_run_dir
from src.gpu import cap_gpu_memory
from src.nas.search import build_encoder
from src.recommender import FixedVectors, NewsRecommender
from src.seed import seed_everything

ARCH = {"channels": 64, "depth": 5, "out_dim": 384}
EPOCHS, DISTILL_EPOCHS = 8, 15
TRAIN_IMPRESSIONS = EVAL_IMPRESSIONS = DISTILL_TITLES = None      # None = everything
STAGES = ("heuristics", "teacher", "student", "nrms")

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("stages", nargs="*", help=f"any of {STAGES}; default all")
parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
parser.add_argument("--smoke", action="store_true",
                    help="tiny settings, two languages, results kept out of paper/results")
args = parser.parse_args()
assert all(s in STAGES for s in args.stages), f"stages must be among {STAGES}"

cfg = load_config()
use_run_dir(cfg, "p1_smoke" if args.smoke else "p1")
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
LANGS = ["en"] + XLANGS
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{(time.time() - T0) / 60:7.1f}m] {msg}", flush=True)


def load_stage(stage: str) -> dict:
    path = RESULTS / f"p1_{stage}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def finished(stage: str, key: str) -> bool:
    """Re-read the stage file each time: another process may be filling it."""
    return key in load_stage(stage)


def save_run(stage: str, key: str, settings: dict, results: dict, started: float) -> None:
    """Record one finished run in the stage file and in the experiment log."""
    record = runlog.append(cfg, f"p1/{stage}/{key}", settings, results, started)
    done = load_stage(stage)
    done[key] = {k: record[k] for k in ("seed", "minutes", "settings", "results")}
    (RESULTS / f"p1_{stage}.json").write_text(json.dumps(done, indent=1), encoding="utf-8")
    log(f"{stage}/{key}: EN AUC {results['en']['all']['auc']:.4f}"
        + (f", mean of 14 languages {results['mean_xlang_auc']:.4f}" if "mean_xlang_auc" in results else ""))


def rounded(d: dict) -> dict:
    return {k: (rounded(v) if isinstance(v, dict) else round(v, 4) if isinstance(v, float) else v)
            for k, v in d.items()}


def language_results(evaluate_lang) -> dict:
    """English by history bucket, then every xMIND language on the full dev set."""
    out = {"en": rounded(evaluate_lang("en", True))}
    for lang in XLANGS:
        out[lang] = rounded(evaluate_lang(lang, False))
    out["mean_xlang_auc"] = round(statistics.fmean(out[l]["auc"] for l in XLANGS), 4)
    return out


def byte_model_results(model) -> dict:
    def evaluate_lang(lang, buckets):
        if buckets:
            return recommender.evaluate_by_history(cfg, model, max_impressions=EVAL_IMPRESSIONS)
        return recommender.evaluate(cfg, model, lang=lang, max_impressions=EVAL_IMPRESSIONS)
    return language_results(evaluate_lang)


# ------------------------------------------------------------------ stages
def stage_heuristics(_seeds) -> None:
    if "heuristics" in load_stage("heuristics"):
        return
    started = time.time()
    res = rounded(heuristics.evaluate_heuristics(cfg))
    record = runlog.append(cfg, "p1/heuristics", {"split": "dev"}, res, started)
    (RESULTS / "p1_heuristics.json").write_text(
        json.dumps({"heuristics": {"minutes": record["minutes"], "results": res}}, indent=1),
        encoding="utf-8")
    for name, r in res.items():
        log(f"heuristics/{name}: AUC {r['all']['auc']:.4f} (no history {r['0']['auc']:.4f})")


def stage_teacher(seeds) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pad = torch.zeros(1, cfg["teacher"]["embed_dim"])
    train_vecs = torch.cat([pad, torch.from_numpy(teacher.build_anchors(cfg, "train")[1])])
    dev_vecs, loaded = {}, {}

    def dev_vectors(lang):
        if lang not in dev_vecs:
            cached = (Path(cfg["paths"]["cache_dir"])
                      / f"teacher_{cfg['data']['mind_size']}_dev_{lang}.npy")
            if lang != "en" and not cached.exists() and "model" not in loaded:
                loaded["model"] = teacher.get_teacher(cfg)
            emb = teacher.build_lang_embeddings(cfg, "dev", lang, model=loaded.get("model"))
            dev_vecs[lang] = torch.cat([pad, torch.from_numpy(emb)])
        return dev_vecs[lang]

    for seed in seeds:
        for project in (False, True):
            key = f"teacher_{'linear' if project else 'frozen'}/seed{seed}"
            if finished("teacher", key):
                continue
            started = time.time()
            cfg["seed"] = seed
            seed_everything(seed)
            model = NewsRecommender(FixedVectors(cfg["teacher"]["embed_dim"], project=project))
            model = recommender.train_recommender(cfg, model=model, epochs=EPOCHS,
                                                  max_train_impressions=TRAIN_IMPRESSIONS,
                                                  fixed_vectors=train_vecs)

            def evaluate_lang(lang, buckets, model=model):
                if buckets:
                    return recommender.evaluate_by_history(cfg, model, news_vectors=dev_vectors("en"),
                                                           max_impressions=EVAL_IMPRESSIONS)
                return recommender.evaluate(cfg, model, lang=lang, news_vectors=dev_vectors(lang),
                                            max_impressions=EVAL_IMPRESSIONS)

            save_run("teacher", key, {"news_vectors": cfg["teacher"]["model"], "trainable_linear": project,
                                      "epochs": EPOCHS, "train_langs": ["en"]},
                     language_results(evaluate_lang), started)
    loaded.clear()
    if device == "cuda":
        torch.cuda.empty_cache()


def stage_student(seeds) -> None:
    ex = torch.zeros(1, cfg["data"]["max_title_bytes"], dtype=torch.long)
    cost = footprint.summarize(build_encoder(cfg, ARCH), ex, cfg, precision="fp32")
    for seed in seeds:
        cfg["seed"] = seed
        distilled = None

        def get_distilled():
            nonlocal distilled
            if distilled is None:
                path = CKPT / f"student_distilled_seed{seed}.pt"
                seed_everything(seed)
                enc = build_encoder(cfg, ARCH)
                if path.exists():
                    enc.load_state_dict(torch.load(path, map_location="cpu"))
                else:
                    enc = student.distill_encoder(cfg, enc, epochs=DISTILL_EPOCHS,
                                                  max_titles=DISTILL_TITLES).cpu()
                    torch.save(enc.state_dict(), path)
                distilled = enc
            return copy.deepcopy(distilled)

        runs = {
            "scratch_en": dict(init="random", train_langs=["en"]),
            "distill_frozen": dict(init="distilled", train_langs=["en"], frozen=True),
            "distill_ft_en": dict(init="distilled", train_langs=["en"]),
            "distill_ft_mixed": dict(init="distilled", train_langs=LANGS),
            "scratch_mixed": dict(init="random", train_langs=LANGS),
        }
        for name, spec in runs.items():
            key = f"{name}/seed{seed}"
            if finished("student", key):
                continue
            started = time.time()
            settings = {"arch": ARCH, "epochs": EPOCHS, "encoder_cost": cost, **spec}
            if spec["init"] == "distilled":
                settings["distill_epochs"] = DISTILL_EPOCHS
            encoder = get_distilled() if spec["init"] == "distilled" else None
            seed_everything(seed)
            if spec.get("frozen"):
                settings["teacher_cosine_dev"] = student.teacher_similarity(cfg, encoder.cuda())
                table = recommender.stack_byte_matrices(cfg, "train", ["en"], "cuda")[1]
                model = NewsRecommender(FixedVectors(ARCH["out_dim"]))
                model = recommender.train_recommender(
                    cfg, model=model, epochs=EPOCHS, max_train_impressions=TRAIN_IMPRESSIONS,
                    fixed_vectors=recommender.encode_titles(encoder, table))

                def evaluate_lang(lang, buckets, encoder=encoder, model=model):
                    vecs = recommender.encode_titles(
                        encoder, recommender.eval_vocab(cfg, "dev", lang).byte_matrix)
                    if buckets:
                        return recommender.evaluate_by_history(cfg, model, news_vectors=vecs,
                                                               max_impressions=EVAL_IMPRESSIONS)
                    return recommender.evaluate(cfg, model, lang=lang, news_vectors=vecs,
                                                max_impressions=EVAL_IMPRESSIONS)

                results = language_results(evaluate_lang)
            else:
                model = recommender.train_recommender(
                    cfg, news_encoder=encoder or build_encoder(cfg, ARCH), epochs=EPOCHS,
                    max_train_impressions=TRAIN_IMPRESSIONS, langs=spec["train_langs"])
                results = byte_model_results(model)
                torch.save(model.state_dict(), CKPT / f"{name}_seed{seed}.pt")
            save_run("student", key, settings, results, started)
            del model
            torch.cuda.empty_cache()


def stage_nrms(seeds) -> None:
    for seed in seeds:
        for variant in ("glove", "reduced"):
            key = f"nrms_{variant}/seed{seed}"
            if finished("nrms", key):
                continue
            started = time.time()
            cfg["seed"] = seed
            seed_everything(seed)
            model = baseline_nrms.train_nrms(cfg, epochs=EPOCHS, variant=variant,
                                             max_train_impressions=TRAIN_IMPRESSIONS)
            params = sum(p.numel() for p in model.parameters())
            results = {"en": rounded(baseline_nrms.evaluate_nrms(
                cfg, model, by_history=True, max_impressions=EVAL_IMPRESSIONS))}
            save_run("nrms", key, {"variant": variant, "epochs": EPOCHS, "params": params,
                                   "embedding_params": model.embed.weight.numel(),
                                   "size_mb_fp32": round(params * 4 / 2 ** 20, 2)}, results, started)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    stages = args.stages or list(STAGES)
    log(f"stages={stages} seeds={args.seeds} langs={len(LANGS)} smoke={args.smoke} device="
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    for stage in stages:
        try:
            globals()[f"stage_{stage}"](args.seeds)
        except Exception:                                  # keep the remaining stages running
            log(f"STAGE {stage} FAILED:\n{traceback.format_exc()}")
    log("P1 DONE")
