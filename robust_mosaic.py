"""#1 — Hardened tile-mosaic pipeline.

VIAME's `tools/create_mosaic.py` consumes pre-computed homographies from the Suppression /
registration step and warps+pastes with them, with NO validation and NO error handling. So
when Suppression emits a degenerate/ill-conditioned/missing homography (AEP: ~50% of mosaics),
the current code silently produces a **distorted mosaic** or drops frames — the exact PWS
symptom ("scripts frequently failed and created distorted mosaics").

This module wraps that step with (a) homography validation, (b) an ORB-geometry recovery
fallback (reusing the #2 utility), and (c) graceful per-frame failure — turning a distortion/
crash into a logged skip + recovery. No new learned model (PWS line 89 compliant).
"""
from __future__ import annotations
import numpy as np
import cv2


def estimate_homography(img_a, img_b):
    """ORB + RANSAC homography mapping frame B into frame A. None if it can't be found."""
    orb = cv2.ORB_create(4000)
    ka, da = orb.detectAndCompute(img_a, None)
    kb, db = orb.detectAndCompute(img_b, None)
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(db, da)
    if len(matches) < 12:
        return None
    matches = sorted(matches, key=lambda m: m.distance)[:400]
    pts_b = np.float32([kb[m.queryIdx].pt for m in matches])
    pts_a = np.float32([ka[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 5.0)
    return H


def _valid_homography(H):
    """Guard against degenerate/ill-conditioned homographies that distort mosaics."""
    if H is None:
        return False
    det = np.linalg.det(H[:2, :2])
    return 0.05 < abs(det) < 20.0 and np.isfinite(H).all()


def validate_and_recover(frames, homographies, pivot=0):
    """Given frames + their homographies (some may be None/degenerate), return a valid
    homography for every frame — validating each, and recovering bad ones geometrically
    against the pivot frame. Returns (recovered_homographies, report)."""
    out, report = [], {"validated": 0, "recovered": 0, "unrecoverable": 0}
    for i, (frame, H) in enumerate(zip(frames, homographies)):
        if _valid_homography(H):
            out.append(H); report["validated"] += 1; continue
        # current code would distort here; instead recover from image geometry
        H_geo = estimate_homography(frames[pivot], frame)
        if _valid_homography(H_geo):
            out.append(H_geo); report["recovered"] += 1
        else:
            out.append(None); report["unrecoverable"] += 1  # skip, don't distort
    return out, report


def safe_mosaic_extent(frames, homographies):
    """Compute the mosaic canvas from only VALID homographies — a degenerate H would otherwise
    blow the canvas up to absurd dimensions (the 'distorted mosaic' failure)."""
    corners = []
    for frame, H in zip(frames, homographies):
        if not _valid_homography(H):
            continue
        h, w = frame.shape[:2]
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners.append(cv2.perspectiveTransform(pts, H).reshape(-1, 2))
    if not corners:
        return None
    allc = np.vstack(corners)
    return allc[:, 0].min(), allc[:, 1].min(), allc[:, 0].max(), allc[:, 1].max()


def _demo():
    rng = np.random.default_rng(3)
    W, H = 800, 700
    scene = cv2.GaussianBlur(rng.integers(0, 255, (H, W + 500), np.uint8), (3, 3), 0)
    frames = [scene[:, i * 250:i * 250 + W] for i in range(3)]  # 3 overlapping frames

    # homographies from "Suppression": frame0 identity, frame1 good translation, frame2 DEGENERATE
    good1 = np.array([[1, 0, -250.0], [0, 1, 0], [0, 0, 1]])
    degenerate = np.array([[1e-6, 0, 0.0], [0, 1e-6, 0], [0, 0, 1]])  # near-singular -> distortion
    homogs = [np.eye(3), good1, degenerate]

    # current create_mosaic.py behavior: use them blindly
    naive_extent = safe_mosaic_extent(frames, homogs)  # (skips invalid here for safety of demo)
    bad_used = not _valid_homography(homogs[2])
    print(f"Suppression homographies: valid={sum(_valid_homography(h) for h in homogs)}/3 "
          f"(frame2 is degenerate -> current code would DISTORT)")

    recovered, report = validate_and_recover(frames, homogs)
    print(f"hardened pipeline -> validated {report['validated']}, "
          f"recovered {report['recovered']} via ORB geometry, unrecoverable {report['unrecoverable']}")
    ext = safe_mosaic_extent(frames, recovered)
    span = (ext[2] - ext[0], ext[3] - ext[1])
    print(f"mosaic extent after recovery: {span[0]:.0f} x {span[1]:.0f} px (bounded, not distorted)")
    assert report["recovered"] == 1 and report["unrecoverable"] == 0, "should recover the bad frame"
    assert span[0] < 4000 and span[1] < 4000, "extent must stay bounded (no distortion blow-up)"
    print("PASS: degenerate Suppression homography detected and recovered; mosaic stays bounded")


if __name__ == "__main__":
    _demo()
