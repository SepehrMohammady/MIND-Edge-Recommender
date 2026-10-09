"""Python port of the FeedWell-Edge on-device learner and ranking score (schedule step 7).

Source: FeedWell-Edge, src/edgeml/localLearningService.js (branch edge-encoder): runContinualLearningStep,
scoreArticleForRanking and their constants. scripts/check_app_learner.py runs the JavaScript file itself
in Node on the same events and checks that both give the same state and scores.

Learning, per event of an article's topic:
  open           delta = 0.8
  read_session   delta = 0.5 + min(dwell / 90, 1.2) + min(scroll / 100, 1.0) (+ 0.5 if completed)
  topic weight   clamp(weight + delta, -20, 20)        (weights only grow; no decay)
  drift          short EMA (alpha 0.3) and long EMA (0.05) of the deltas, divergence = |short - long|;
                 a topic is flagged when it has 5 or more events and divergence >= 0.35 (top 5 by divergence)
Score of an article:
  topic weight + drift boost (flagged: max(0.4, 3 x divergence)) + freshness (1.5 - min(hours / 48, 1.5),
  at least 0) + unread (0.35 if not read)
The app rounds the score to 4 decimals (toFixed(4)); so does this port.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SHORT_ALPHA, LONG_ALPHA, THRESHOLD, MIN_EVENTS = 0.3, 0.05, 0.35, 5


def _round4(x: float) -> float:
    # Number(x.toFixed(4)) in JavaScript rounds the exact binary value of x, ties to the larger value
    from decimal import ROUND_HALF_UP, Decimal
    return float(Decimal(x).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


@dataclass
class TopicSignal:
    short_ema: float = 0.0
    long_ema: float = 0.0
    divergence: float = 0.0
    event_count: int = 0


@dataclass
class AppLearner:
    weights: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)
    flagged: list = field(default_factory=list)

    @staticmethod
    def delta(event: dict) -> float:
        if event["type"] == "open":
            return 0.8
        if event["type"] == "read_session":
            d = 0.5 + min(event.get("dwellSeconds", 0) / 90, 1.2) + min(event.get("maxScrollPercent", 0) / 100, 1.0)
            return d + 0.5 if event.get("completed") else d
        return 0.0

    def learn(self, events: list[dict]) -> None:
        """One learning step over new events (the app's runContinualLearningStep from its cursor)."""
        for evt in events:
            d = self.delta(evt)
            if d == 0:
                continue
            topic = evt.get("topic") or "unknown"
            self.weights[topic] = max(-20.0, min(20.0, self.weights.get(topic, 0.0) + d))
            s = self.signals.setdefault(topic, TopicSignal())
            s.short_ema = SHORT_ALPHA * d + (1 - SHORT_ALPHA) * s.short_ema
            s.long_ema = LONG_ALPHA * d + (1 - LONG_ALPHA) * s.long_ema
            s.divergence = abs(s.short_ema - s.long_ema)
            s.event_count += 1
        ranked = sorted(((t, s) for t, s in self.signals.items()
                         if s.event_count >= MIN_EVENTS and s.divergence >= THRESHOLD),
                        key=lambda ts: -ts[1].divergence)
        self.flagged = [t for t, _ in ranked][:5]

    def score(self, topic: str, hours_since_published: float | None, is_read: bool) -> float:
        weight = self.weights.get(topic, 0.0)
        divergence = self.signals[topic].divergence if topic in self.signals else 0.0
        drift = max(0.4, divergence * 3) if topic in self.flagged else 0.0
        hours = 72.0 if hours_since_published is None else max(0.0, hours_since_published)
        fresh = max(0.0, 1.5 - min(hours / 48, 1.5))
        unread = 0.0 if is_read else 0.35
        return _round4(weight + drift + fresh + unread)
