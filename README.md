# Steller Sea Lion — AI Counting Enhancements (capability demonstration)

Unidatum Integrated Products LLC — hands-on demonstration for NOAA/AEP's Steller Sea Lion AI
Image Processing & Counting Enhancement Effort (Solicitation ACQ-AONCASAT01-25-0097).

**Grounding:** built against the real open-source stack — Kitware **DIVE** (Vue/TS + Python),
**VIAME** (`tools/create_mosaic.py`, `register_using_homographies.pipe`,
`configs/add-ons/sea-lion/*`). Compliant with the PWS constraint (line 89): *no new
detection/classification/registration models* — these are **utilities and "further
development" of the existing FUSION/Suppression pipeline**.

**Data & scope:** the AEP aerial imagery, labeled training data, ground-truth counts, and model
weights are NOAA-internal. Items that need that data are marked *[needs AEP data]* and are
demonstrated here on synthetic data / as schema-ready code; on real data the same code runs
against actual detections and the survey's ~60% vertical / ~10% horizontal overlap geometry.

## The problem (AEP's own assessment, PWS §2.2)
Tile-mosaic scripts "frequently failed" / distorted; the Suppression (registration) model
failed on ~50% of mosaics; **93% of missed sea lions trace to Suppression errors** → overlap
isn't de-duplicated → double-counts and misses. This is a **pipeline-robustness + registration
problem**, not a "need a smarter model" problem.

## Enhancements (each on its own branch = "PR")
| # | Enhancement | Branch | Status |
|---|---|---|---|
| 2 | Geometry-based cross-frame dedup / registration fallback | `feat/2-crossframe-dedup` | ✅ runnable, synthetic test passes |
| 1 | Hardened tile-mosaic pipeline (homography validation + graceful failure) | `feat/1-robust-mosaic` | ▶ grounded in `create_mosaic.py` review |
| 4 | QC & accuracy analytics (per-class confusion, per-mosaic suppression rate) | `feat/4-qc-analytics` | ▶ schema-driven, synthetic-runnable |
| 3 | FUSION further-development: fine-tune + class-balancing | — | *[needs AEP data + GPU]* |
| 5 | DIVE human-in-the-loop review prioritization | `feat/5-dive-hitl` | ▶ DIVE (Vue/TS) feature branch |

These are Unidatum's demonstration artifacts, not upstream PRs to Kitware.
