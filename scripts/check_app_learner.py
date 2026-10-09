"""Check the Python port of the FeedWell-Edge learner (src/app_learner.py) against the app's own JavaScript.

A seeded stream of 600 events (opens, reading sessions with dwell and scroll, and events the learner
ignores) over 15 topics is split into segments; after each segment both run one learning step: the app's
src/edgeml/localLearningService.js executed in Node (AsyncStorage replaced by an in-memory mock, clock
fixed), and the Python port. Then 200 articles (topic, hours since publication, read or not) are scored by
both. Topic weights, drift signals, flagged topics and scores must agree.

    python -m scripts.check_app_learner [path/to/localLearningService.js]
Writes paper/results/app_learner_check.json.
"""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from src.app_learner import AppLearner

ROOT = Path(__file__).resolve().parents[1]
JS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Projects/PhD/FeedWell-Edge/src/edgeml/localLearningService.js")
NOW = 1_760_000_000_000                                      # fixed clock (ms) for both sides

rng = random.Random(7)
topics = [f"topic{i}" for i in range(15)]
segments = []
for _ in range(30):
    seg = []
    for _ in range(20):
        t = rng.choice(topics[:rng.randint(3, 15)])
        kind = rng.choices(["open", "read_session", "impression", "search"], [5, 3, 4, 1])[0]
        evt = {"type": kind, "at": "2026-10-09T10:00:00.000Z", "article": {"id": rng.randint(1, 10**6), "topic": t}}
        if kind == "read_session":
            dwell, scroll = rng.randint(0, 300), rng.randint(0, 100)
            evt["context"] = {"dwellSeconds": dwell, "maxScrollPercent": scroll,
                              "completed": scroll >= 80 or dwell >= 45}
        seg.append(evt)
    segments.append(seg)
articles = [{"topic": rng.choice(topics + ["never-seen"]), "hours": rng.choice([None, rng.uniform(0, 100)]),
             "isRead": rng.random() < 0.3} for _ in range(200)]

harness = r"""
const fs = require('fs');
const store = {};
const AsyncStorage = { getItem: async (k) => (k in store ? store[k] : null), setItem: async (k, v) => { store[k] = v; } };
const NOW = %d;
Date.now = () => NOW;
let src = fs.readFileSync(process.argv[2], 'utf8');
src = src.replace(/^import AsyncStorage from .*$/m, '');
src = src.replace(/^export (async )?function/mg, (m, a) => (a ? 'async function' : 'function'));
const mod = new Function('AsyncStorage', src + '\nreturn { runContinualLearningStep, scoreArticleForRanking, getFeedRankingProfile };')(AsyncStorage);
const input = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
(async () => {
  let events = [];
  for (const seg of input.segments) {
    events = events.concat(seg);
    store['edgeml_events_v1'] = JSON.stringify(events);
    await mod.runContinualLearningStep();
  }
  const state = JSON.parse(store['edgeml_model_state_v1']);
  const profile = await mod.getFeedRankingProfile();
  const scores = input.articles.map((a) => mod.scoreArticleForRanking({
    categories: [a.topic], isRead: a.isRead,
    publishedDate: a.hours === null ? undefined : new Date(NOW - a.hours * 3600 * 1000).toISOString() }, profile));
  process.stdout.write(JSON.stringify({ state, scores }));
})();
""" % NOW

with tempfile.TemporaryDirectory() as d:
    (Path(d) / "harness.js").write_text(harness, encoding="utf-8")
    (Path(d) / "input.json").write_text(json.dumps({"segments": segments, "articles": articles}), encoding="utf-8")
    out = subprocess.run(["node", str(Path(d) / "harness.js"), str(JS), str(Path(d) / "input.json")],
                         capture_output=True, text=True, check=True).stdout
js = json.loads(out)

py = AppLearner()
for seg in segments:
    py.learn([{"type": e["type"], "topic": e["article"]["topic"], **e.get("context", {})} for e in seg])
# the JS file gets the hours through ISO dates with millisecond precision: give Python the same hours
hours = [None if a["hours"] is None else round(a["hours"] * 3600 * 1000) / 3600 / 1000 for a in articles]
py_scores = [py.score(a["topic"].lower(), h, a["isRead"]) for a, h in zip(articles, hours)]

w_js = js["state"]["topicWeights"]
sig_js = js["state"]["drift"]["topicSignals"]
diff = {
    "topics": sorted(set(w_js) ^ set(py.weights)),
    "weight_max_abs": max(abs(w_js[t] - py.weights[t]) for t in w_js),
    "divergence_max_abs": max(abs(sig_js[t]["divergence"] - py.signals[t].divergence) for t in sig_js),
    "event_count_mismatch": [t for t in sig_js if sig_js[t]["eventCount"] != py.signals[t].event_count],
    "flagged_js": js["state"]["drift"]["flaggedTopics"], "flagged_py": py.flagged,
    "score_mismatches": sum(1 for a, b in zip(js["scores"], py_scores) if a != b),
    "score_max_abs": max(abs(a - b) for a, b in zip(js["scores"], py_scores)),
}
ok = (not diff["topics"] and diff["weight_max_abs"] < 1e-9 and diff["divergence_max_abs"] < 1e-9
      and not diff["event_count_mismatch"] and diff["flagged_js"] == diff["flagged_py"] and diff["score_mismatches"] == 0)
result = {"js_file": str(JS), "events": sum(len(s) for s in segments), "learning_steps": len(segments),
          "articles_scored": len(articles), **diff, "pass": ok}
(ROOT / "paper/results/app_learner_check.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
print(json.dumps(result, indent=1))
sys.exit(0 if ok else 1)
