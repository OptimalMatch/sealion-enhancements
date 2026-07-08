"""#5 — Human-in-the-loop review prioritization (backend for a DIVE review-queue panel).

Biologists can't re-check every detection. This ranks detections by how likely a human review
is to *fix a count error*, so limited review time targets the animals most likely wrong —
cutting the >5% reconciliation revisits. Priority combines:
  - low model confidence,
  - overlap ambiguity (near a frame seam / in an unsuppressed overlap region — where the
    Suppression failures cause double-counts/misses),
  - class rarity/difficulty (pups + dead classes, which are rare and error-prone).

Runnable engine here; it feeds a DIVE (Vue/TS) "Review Queue" panel — see DIVE_INTEGRATION.md.
No new model (PWS line 89 compliant): pure post-processing over existing FUSION outputs.
"""
from __future__ import annotations
import numpy as np

HARD_CLASSES = {"pup", "dead_pup", "dead_non_pup"}


def priority(det, frame_w, overlap_frac=0.10):
    """0..1 review priority. det: {class, confidence, x, in_overlap(optional)}."""
    conf = det.get("confidence", 1.0)
    seam = overlap_frac * frame_w
    near_seam = 1.0 if (det["x"] < seam or det["x"] > frame_w - seam or det.get("in_overlap")) else 0.0
    rarity = 1.0 if det["class"] in HARD_CLASSES else 0.0
    # weighted: confidence dominates, ambiguity + rarity escalate
    return round(0.6 * (1 - conf) + 0.25 * near_seam + 0.15 * rarity, 3)


def review_queue(dets, frame_w, top_frac=0.2):
    ranked = sorted(dets, key=lambda d: priority(d, frame_w), reverse=True)
    for d in ranked:
        d["review_priority"] = priority(d, frame_w)
    k = max(1, int(len(ranked) * top_frac))
    return ranked, ranked[:k]


def _demo():
    rng = np.random.default_rng(5)
    W, classes = 1000, ["adult_female", "juvenile", "adult_male", "pup", "dead_pup"]
    dets = [{"class": rng.choice(classes), "confidence": float(rng.uniform(.35, .99)),
             "x": float(rng.uniform(0, W)), "in_overlap": bool(rng.random() < 0.2)}
            for _ in range(200)]
    ranked, queue = review_queue(dets, W, top_frac=0.2)
    print(f"{len(dets)} detections -> review queue of {len(queue)} (top 20% by priority)")
    # the queue should over-represent low-confidence / seam / hard-class detections
    q_lowconf = np.mean([d["confidence"] for d in queue])
    a_lowconf = np.mean([d["confidence"] for d in dets])
    q_hard = np.mean([d["class"] in HARD_CLASSES for d in queue])
    a_hard = np.mean([d["class"] in HARD_CLASSES for d in dets])
    print(f"mean confidence: queue {q_lowconf:.2f} vs all {a_lowconf:.2f} (queue targets low-conf)")
    print(f"hard-class share: queue {q_hard:.0%} vs all {a_hard:.0%} (queue targets rare classes)")
    print("top of queue:")
    for d in queue[:4]:
        print(f"  prio {d['review_priority']:.2f}  {d['class']:14} conf {d['confidence']:.2f}"
              f"{'  [overlap]' if d.get('in_overlap') else ''}")
    assert q_lowconf < a_lowconf and q_hard >= a_hard
    print("PASS: review queue concentrates the likely-wrong detections for the biologist")


if __name__ == "__main__":
    _demo()
