# #5 — DIVE integration for HITL review prioritization

`review_prioritization.py` is the engine; it surfaces the detections most worth a human's time.
In DIVE (Kitware's Vue 2 / TypeScript client + Girder Python server) it plugs in as:

**Server (Girder plugin, Python)** — an endpoint that runs `review_queue()` over a dataset's
FUSION detections and returns them ranked by `review_priority` (with the reasons: low
confidence / seam-overlap / hard class).

**Client (Vue/TS)** — a "Review Queue" side panel component in `client/dive-common/` that:
- lists detections sorted by `review_priority` (highest first),
- on click, seeks the annotation viewer to that detection's frame + selects the box,
- lets the biologist confirm / reclassify / delete, then advances to the next,
- shows a progress bar ("reviewed 18 / 40 high-priority").

This is standard DIVE feature work (a panel + a Girder endpoint) — Unidatum's exact stack
(Vue/TS + Python) — and needs a running DIVE instance (`docker compose up`) + a dataset to
demo end-to-end. No new model: it only re-orders existing FUSION outputs (PWS line 89
compliant).
