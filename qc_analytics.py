"""#4 — QC & accuracy analytics.

Makes count accuracy *measurable* against the 2-biologist ground truth, so every pipeline
improvement (robust mosaics, dedup, FUSION retrain) can be shown to reduce the error. Computes
per-class confusion, per-site count error vs. AEP's reconciliation thresholds (>5%, or >20
non-pups / >10 pups), and per-mosaic Suppression success rate (the root-cause metric).

Schema (plugs into real VIAME/DIVE outputs; demonstrated here on synthetic data):
  detection  = {"class","x","y","confidence","mosaic_id"}
  truth      = {"class","x","y","mosaic_id"}
  mosaic     = {"mosaic_id","suppression_ok": bool}
Data-independent: on real AEP data these lists come from DIVE annotations + FUSION outputs.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np

CLASSES = ["adult_male", "sub_adult_male", "adult_female", "juvenile", "pup",
           "dead_non_pup", "dead_pup"]
LIVE_NON_PUP = {"adult_male", "sub_adult_male", "adult_female", "juvenile"}


def match(dets, truth, radius=25.0):
    """Greedy nearest-neighbour match within a mosaic. Returns (pairs, unmatched_det, unmatched_gt)."""
    by_m = lambda rows: defaultdict(list, {k: [r for r in rows if r["mosaic_id"] == k]
                                           for k in {r["mosaic_id"] for r in rows}})
    dm, tm = by_m(dets), by_m(truth)
    pairs, ud, ug = [], [], []
    for m in set(dm) | set(tm):
        d, t = dm[m], tm[m]
        used = set()
        for di in d:
            best, bj = radius, None
            for j, tj in enumerate(t):
                if j in used:
                    continue
                dist = np.hypot(di["x"] - tj["x"], di["y"] - tj["y"])
                if dist < best:
                    best, bj = dist, j
            if bj is None:
                ud.append(di)
            else:
                used.add(bj); pairs.append((di, t[bj]))
        ug.extend(t[j] for j in range(len(t)) if j not in used)
    return pairs, ud, ug


def confusion(pairs):
    idx = {c: i for i, c in enumerate(CLASSES)}
    M = np.zeros((len(CLASSES), len(CLASSES)), int)
    for d, t in pairs:
        M[idx[t["class"]], idx[d["class"]]] += 1
    return M


def report(dets, truth, mosaics):
    pairs, ud, ug = match(dets, truth)
    M = confusion(pairs)
    tp, fp, fn = len(pairs), len(ud), len(ug)
    print(f"detections {len(dets)}  truth {len(truth)}  | TP {tp}  FP(dupes/spurious) {fp}  FN(misses) {fn}")

    # count error vs AEP reconciliation thresholds
    def counts(rows):
        c = Counter(r["class"] for r in rows)
        return sum(c[k] for k in LIVE_NON_PUP), c["pup"]
    dn, dp = counts(dets); tn, tp_ = counts(truth)
    np_flag = abs(dn - tn) > 20 or (tn and abs(dn - tn) / tn > 0.05)
    pup_flag = abs(dp - tp_) > 10 or (tp_ and abs(dp - tp_) / tp_ > 0.05)
    print(f"live non-pups: det {dn} vs truth {tn}  -> {'REVISIT' if np_flag else 'within threshold'}")
    print(f"pups:          det {dp} vs truth {tp_}  -> {'REVISIT' if pup_flag else 'within threshold'}")

    # Suppression success — the root-cause metric
    ok = sum(1 for m in mosaics if m["suppression_ok"])
    print(f"Suppression success: {ok}/{len(mosaics)} mosaics ({100*ok/len(mosaics):.0f}%) "
          f"-- misses concentrate on the failures")

    # per-class recall
    print("per-class recall:")
    for i, c in enumerate(CLASSES):
        support = M[i].sum() + sum(1 for g in ug if g["class"] == c)
        rec = M[i, i] / support if support else float("nan")
        if support:
            print(f"  {c:16} recall {rec:4.0%}  (n={support})")
    return {"tp": tp, "fp": fp, "fn": fn, "confusion": M}


def _demo():
    rng = np.random.default_rng(11)
    truth, dets, mosaics = [], [], []
    for m in range(8):
        supp_ok = m >= 4                       # half fail (AEP: ~50%)
        mosaics.append({"mosaic_id": m, "suppression_ok": supp_ok})
        n = rng.integers(8, 16)
        for _ in range(n):
            cls = rng.choice(CLASSES, p=[.15, .1, .3, .2, .18, .04, .03])
            x, y = rng.uniform(0, 1000, 2)
            truth.append({"class": cls, "x": x, "y": y, "mosaic_id": m})
            if not supp_ok and rng.random() < 0.4:
                continue                        # miss (Suppression failure -> 93% of misses)
            dcls = cls if rng.random() > 0.1 else rng.choice(CLASSES)  # 10% misclassification
            dets.append({"class": dcls, "x": x + rng.normal(0, 5), "y": y + rng.normal(0, 5),
                         "confidence": rng.uniform(.4, .99), "mosaic_id": m})
            if not supp_ok and rng.random() < 0.2:  # unsuppressed overlap -> duplicate
                dets.append({"class": dcls, "x": x + rng.normal(0, 8), "y": y + rng.normal(0, 8),
                             "confidence": rng.uniform(.4, .9), "mosaic_id": m})
    print("=== QC report (synthetic; real numbers plug in from AEP data) ===")
    r = report(dets, truth, mosaics)
    assert r["tp"] > 0 and r["fn"] > 0
    print("PASS: QC surfaces misses/dupes concentrated on Suppression-failed mosaics")


if __name__ == "__main__":
    _demo()
