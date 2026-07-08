"""#3 — FUSION "further development": fine-tune + class-balancing (SCAFFOLD — NOT RUN HERE).

Honest status: this REQUIRES the AEP training imagery + labels and a GPU, which are
NOAA-internal — so it is a *structured scaffold* showing the approach, not an executed run. It
does NOT train a new model (PWS line 89): it continues training the EXISTING FUSION detector
(VIAME `configs/add-ons/sea-lion/tracker_sea_lion_tracker_fusion_all_class.pipe`, an
Ultralytics/YOLO-family model per `plugins/pytorch/ultralytics_trainer.py`).

Approach encoded below:
  1. Load AEP dataset (DIVE/VIAME annotation export) split by site.
  2. Class-balanced sampling — up-weight rare/hard classes (pup, dead_pup, dead_non_pup) that
     drive count error, so the detector stops missing them.
  3. Hard-negative mining — mine false positives (water, rocks) from prior runs as negatives.
  4. Fine-tune from the existing FUSION weights (transfer learning; low LR; freeze backbone
     early), NOT from scratch.
  5. Evaluate against held-out sites with the QC metrics in qc_analytics.py; ship only if
     per-class recall (esp. pups) improves without regressing others.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass, field

RARE_HARD = ["pup", "dead_pup", "dead_non_pup"]


@dataclass
class FinetuneConfig:
    base_weights: str = "<AEP FUSION weights>.pt"     # existing model — do NOT start from scratch
    data_yaml: str = "<AEP dataset>.yaml"             # DIVE/VIAME export
    epochs: int = 40
    lr0: float = 1e-3                                  # low LR for fine-tuning
    freeze_backbone_epochs: int = 5
    class_weights: dict = field(default_factory=lambda: {c: 3.0 for c in RARE_HARD})
    hard_negative_dir: str = "<mined FPs: water/rock>"


def build_sampler(labels, class_weights):
    """Weighted sampler so rare/hard classes appear more often per epoch (approach stub)."""
    weights = [max(class_weights.get(c, 1.0) for c in img_classes) for img_classes in labels]
    return weights  # -> torch.utils.data.WeightedRandomSampler(weights, len(weights))


def finetune(cfg: FinetuneConfig):
    try:
        from ultralytics import YOLO  # VIAME's FUSION is Ultralytics-family
    except ImportError:
        sys.exit("[#3] ultralytics + GPU + AEP data required — scaffold only; not executed here.")
    model = YOLO(cfg.base_weights)                    # continue existing model, not new
    model.train(data=cfg.data_yaml, epochs=cfg.epochs, lr0=cfg.lr0,
                freeze=cfg.freeze_backbone_epochs)     # class-balancing via sampler/loss weights
    return model


if __name__ == "__main__":
    print(__doc__)
    print("Config that WOULD run against AEP data:")
    print(" ", FinetuneConfig())
    print("\nNOT EXECUTED — requires AEP imagery/labels + GPU. Structure + approach only.")
