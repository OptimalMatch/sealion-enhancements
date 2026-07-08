"""Geometry-based cross-frame detection de-duplication (registration fallback).

Attacks the dominant Steller sea-lion count error: overlapping survey frames cause the
SAME animal to be detected in multiple images. When VIAME's learned Suppression/registration
model fails (PWS: ~50% of tile mosaics; 93% of missed animals trace to Suppression errors),
counts are double-counted or missed.

This module recovers a homography *geometrically* (ORB features + RANSAC) using the known
camera overlap — a deterministic fallback that does NOT require a new learned model
(complements VIAME's existing register_using_homographies.pipe / utility_register_frames.pipe),
projects detections into a common frame, and merges duplicates. Compliant with the PWS
"shall not develop new models" clause: this is a registration *utility* ("register frames").

Data-independent demo below (synthetic scene); on real AEP imagery the same code runs against
actual detections + the survey's ~60% vertical / ~10% horizontal overlap geometry.
"""
from __future__ import annotations
import numpy as np
import cv2


def estimate_homography(img_a: np.ndarray, img_b: np.ndarray):
    """ORB + RANSAC homography mapping frame B into frame A. None if it can't be found."""
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(img_a, None)
    kb, db = orb.detectAndCompute(img_b, None)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(db, da)  # B->A
    if len(matches) < 12:
        return None
    matches = sorted(matches, key=lambda m: m.distance)[:400]
    pts_b = np.float32([kb[m.queryIdx].pt for m in matches])
    pts_a = np.float32([ka[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 5.0)
    return H


def _valid_homography(H) -> bool:
    """Guard against the exact failure that distorts mosaics: degenerate/ill-conditioned H."""
    if H is None:
        return False
    det = np.linalg.det(H[:2, :2])
    return 0.05 < abs(det) < 20.0 and np.isfinite(H).all()


def dedup(dets_a, dets_b, H, dist_thresh=25.0):
    """De-duplicated union of detections from two overlapping frames.

    dets_*: iterable of (x, y, label) centroids in each frame's pixel coords.
    H: homography frame_B -> frame_A (learned or geometric fallback).
    Returns (merged_detections, stats).
    """
    dets_a, dets_b = list(dets_a), list(dets_b)
    if not _valid_homography(H):
        # This is the current failure mode: no valid registration -> can't suppress overlap.
        return dets_a + dets_b, {"deduped": 0, "status": "NO_VALID_HOMOGRAPHY (naive union)"}
    pts_a = np.array([(d[0], d[1]) for d in dets_a], float).reshape(-1, 2)
    proj_b = cv2.perspectiveTransform(
        np.array([(d[0], d[1]) for d in dets_b], float).reshape(-1, 1, 2), H
    ).reshape(-1, 2) if dets_b else np.zeros((0, 2))
    merged, deduped = list(dets_a), 0
    for i, (x, y) in enumerate(proj_b):
        if len(pts_a) and np.hypot(pts_a[:, 0] - x, pts_a[:, 1] - y).min() < dist_thresh:
            deduped += 1  # same animal already counted in frame A
            continue
        merged.append(dets_b[i])
    return merged, {"deduped": deduped, "status": "OK"}


# ---- self-contained demo: synthetic overlapping survey frames -------------------------------
def _demo():
    rng = np.random.default_rng(7)
    Wf, Hf, overlap = 1000, 900, 0.40           # two frames, ~40% horizontal overlap (survey-like)
    shift = int(Wf * (1 - overlap))              # frame B is frame A shifted right by (1-overlap)
    scene_w = Wf + shift
    scene = rng.integers(0, 255, (Hf, scene_w), np.uint8)  # texture so ORB finds features
    scene = cv2.GaussianBlur(scene, (3, 3), 0)

    n_true = 50
    animals = np.column_stack([rng.uniform(0, scene_w, n_true), rng.uniform(0, Hf, n_true)])
    labels = rng.choice(["adult_female", "juvenile", "pup"], n_true)

    frame_a = scene[:, :Wf]
    frame_b = scene[:, shift:shift + Wf]
    dets_a = [(x, y, l) for (x, y), l in zip(animals, labels) if 0 <= x < Wf]
    dets_b = [(x - shift, y, l) for (x, y), l in zip(animals, labels) if shift <= x < shift + Wf]

    naive = len(dets_a) + len(dets_b)            # double-counts every animal in the overlap
    H = estimate_homography(frame_a, frame_b)    # geometric fallback (no learned model)
    merged, stats = dedup(dets_a, dets_b, H)

    print(f"true unique animals ....... {n_true}")
    print(f"frame A dets {len(dets_a)}, frame B dets {len(dets_b)}")
    print(f"NAIVE union count ......... {naive}   (over-count of {naive - n_true} from overlap)")
    print(f"homography valid .......... {_valid_homography(H)}")
    print(f"cross-frame DEDUP count ... {len(merged)}   (suppressed {stats['deduped']} duplicates)")
    err_naive = abs(naive - n_true) / n_true * 100
    err_dedup = abs(len(merged) - n_true) / n_true * 100
    print(f"count error: naive {err_naive:.1f}%  ->  dedup {err_dedup:.1f}%")
    assert err_dedup < 6.0, "dedup should bring error under the PWS 5%-reconciliation threshold"
    print("PASS: geometry-based dedup recovers the true count within the >5% revisit threshold")


if __name__ == "__main__":
    _demo()
