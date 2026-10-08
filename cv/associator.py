"""
associator.py

Person-object association logic for the Hacknex CV pipeline.

This module is intentionally separate from pipeline.py so that the
backend / RAG team can import association results without pulling in
OpenCV or the pipeline loop.

Public API
──────────
    ObjectCandidate        dataclass – a single object detection tied to a person
    ObjectTrackBuffer      per-camera temporal history of (person, class) pairs
    calculate_spatial_score(person_bbox, object_bbox)  -> float  [0,1]
    calculate_seg_score(person_mask, object_mask)      -> float  [0,1]
    calculate_evidence_score(...)                      -> float  [0,1]
    associate_object_with_person(person_bbox, object_bbox,
                                 person_mask, object_mask)
                                                       -> dict with scores

Association states
──────────────────
    "candidate"  – spatially plausible, low or unconfirmed evidence
    "probable"   – moderate evidence; worth investigating
    "confirmed"  – strong, temporally consistent evidence

Only "confirmed" (and optionally "probable") associations are written to the
final JSON output and used to generate evidence frames.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config import (
    CONFIRMED_ASSOCIATION_THRESHOLD,
    CONFIDENCE_WEIGHT,
    MAX_ASSOC_DISTANCE_FACTOR,
    MIN_OBJECT_CONFIRM_FRAMES,
    PROBABLE_ASSOCIATION_THRESHOLD,
    PROBABLE_FRAMES,
    SEG_WEIGHT,
    SPATIAL_WEIGHT,
    TEMPORAL_WEIGHT,
)

# ── Types ─────────────────────────────────────────────────────────────────────
BBox = List[int]   # [x1, y1, x2, y2]


# ── Dataclass ─────────────────────────────────────────────────────────────────
@dataclass
class ObjectCandidate:
    """One detected object that has passed the initial spatial check for a person."""
    class_name: str
    confidence: float
    bbox: BBox                       # full-frame coordinates
    person_track_id: int
    spatial_score: float  = 0.0
    seg_score: float      = 0.0      # 0 if segmentation not available
    temporal_score: float = 0.0      # filled in by ObjectTrackBuffer
    evidence_score: float = 0.0      # final composite score
    association_state: str = "candidate"
    segmentation_available: bool = False
    source: str = "fullframe"        # "fullframe" | "crop"

    def to_dict(self) -> dict:
        return {
            "class_name":             self.class_name,
            "detection_confidence":   round(self.confidence, 3),
            "bbox":                   self.bbox,
            "spatial_score":          round(self.spatial_score, 3),
            "seg_score":              round(self.seg_score, 3),
            "temporal_score":         round(self.temporal_score, 3),
            "evidence_score":         round(self.evidence_score, 3),
            "association_state":      self.association_state,
            "segmentation_available": self.segmentation_available,
            "source":                 self.source,
        }


# ── Spatial scoring ───────────────────────────────────────────────────────────
def _box_centre(bbox: BBox) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _box_dims(bbox: BBox) -> Tuple[float, float]:
    return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])


def _iou(a: BBox, b: BBox) -> float:
    """Intersection over Union of two boxes."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _point_in_box(px: float, py: float, bbox: BBox) -> bool:
    return bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]


def calculate_spatial_score(person_bbox: BBox,
                             object_bbox: BBox,
                             crop_padding: float = 0.20) -> float:
    """
    Compute a spatial association score in [0, 1].

    Scoring logic (additive, then normalised):
      +1.00  object centre is inside the person bbox
      +0.80  object centre is inside the padded person bbox
      +0.40  any overlap (positive IoU) with the person bbox
      +0.20  object centre is within MAX_ASSOC_DISTANCE_FACTOR * person_dim
             of the nearest edge of the person bbox

    A score of 0 means "too far away – do not associate at all."
    """
    pw, ph = _box_dims(person_bbox)
    person_dim = max(pw, ph, 1.0)

    # Padded person bbox (same logic as crops.padded_box)
    pad_x = pw * crop_padding
    pad_y = ph * crop_padding
    padded = [
        person_bbox[0] - pad_x,
        person_bbox[1] - pad_y,
        person_bbox[2] + pad_x,
        person_bbox[3] + pad_y,
    ]

    ocx, ocy = _box_centre(object_bbox)

    raw = 0.0

    # 1) Object centre inside strict person box
    if _point_in_box(ocx, ocy, person_bbox):
        raw += 1.00

    # 2) Object centre inside padded person box (but not strict)
    elif _point_in_box(ocx, ocy, padded):
        raw += 0.80

    # 3) Any overlap
    iou = _iou(person_bbox, object_bbox)
    if iou > 0:
        raw += 0.40 + 0.20 * iou   # proportional bonus for strong overlap

    # 4) Distance from object centre to nearest point on person bbox
    clamp_x = max(person_bbox[0], min(ocx, person_bbox[2]))
    clamp_y = max(person_bbox[1], min(ocy, person_bbox[3]))
    dist = math.hypot(ocx - clamp_x, ocy - clamp_y)
    normalised_dist = dist / person_dim

    if normalised_dist > MAX_ASSOC_DISTANCE_FACTOR:
        return 0.0   # definitely not associated

    if raw == 0.0:
        # Object is outside but close: partial score
        raw = max(0.0, 0.20 * (1.0 - normalised_dist / MAX_ASSOC_DISTANCE_FACTOR))

    return min(raw, 1.0)


# ── Segmentation scoring ──────────────────────────────────────────────────────
def calculate_seg_score(person_mask, object_mask) -> float:
    """
    Estimate association strength from segmentation masks.

    person_mask / object_mask : numpy bool arrays (H×W) OR None.
    Returns 0.0 if either mask is None (segmentation unavailable).

    Scoring:
      - Mask overlap (fraction of object mask that overlaps with person mask)
        → high overlap suggests the object is on/touching the person
      - Proximity bonus: distance between mask centroids
    """
    if person_mask is None or object_mask is None:
        return 0.0

    try:
        import numpy as np

        # Boolean arrays
        pm = person_mask.astype(bool)
        om = object_mask.astype(bool)

        obj_area = om.sum()
        if obj_area == 0:
            return 0.0

        intersection = (pm & om).sum()
        overlap_ratio = intersection / obj_area  # fraction of object covered by person

        # Centroid distance (normalised to image diagonal)
        h, w = pm.shape
        diag = math.hypot(h, w) or 1.0

        def centroid(mask):
            ys, xs = np.where(mask)
            if len(xs) == 0:
                return w / 2, h / 2
            return xs.mean(), ys.mean()

        pcx, pcy = centroid(pm)
        ocx, ocy = centroid(om)
        dist_norm = math.hypot(pcx - ocx, pcy - ocy) / diag
        proximity = max(0.0, 1.0 - dist_norm * 3.0)   # 0 at ~1/3 diagonal away

        score = 0.7 * overlap_ratio + 0.3 * proximity
        return float(min(score, 1.0))

    except Exception:
        return 0.0


# ── Evidence score ────────────────────────────────────────────────────────────
def calculate_evidence_score(spatial_score: float,
                              temporal_score: float,
                              detection_confidence: float,
                              seg_score: float = 0.0,
                              segmentation_available: bool = False) -> float:
    """
    Compose a normalised evidence score in [0, 1].

    Weights (from config.py):
        SPATIAL_WEIGHT    = 0.35
        TEMPORAL_WEIGHT   = 0.40
        CONFIDENCE_WEIGHT = 0.10
        SEG_WEIGHT        = 0.15

    When segmentation is unavailable, its weight is redistributed equally to
    the remaining three components so the total still sums to 1.0.
    """
    if not segmentation_available or seg_score <= 0.0:
        # Redistribute seg weight
        total_other = SPATIAL_WEIGHT + TEMPORAL_WEIGHT + CONFIDENCE_WEIGHT
        w_sp = SPATIAL_WEIGHT    / total_other
        w_te = TEMPORAL_WEIGHT   / total_other
        w_co = CONFIDENCE_WEIGHT / total_other
        score = (w_sp * spatial_score
                 + w_te * temporal_score
                 + w_co * detection_confidence)
    else:
        score = (SPATIAL_WEIGHT    * spatial_score
                 + TEMPORAL_WEIGHT   * temporal_score
                 + CONFIDENCE_WEIGHT * detection_confidence
                 + SEG_WEIGHT        * seg_score)

    return float(min(max(score, 0.0), 1.0))


def association_state_from_score(evidence_score: float) -> str:
    """Map evidence score to a human-readable state."""
    if evidence_score >= CONFIRMED_ASSOCIATION_THRESHOLD:
        return "confirmed"
    if evidence_score >= PROBABLE_ASSOCIATION_THRESHOLD:
        return "probable"
    return "candidate"


# ── Full association entry point ──────────────────────────────────────────────
def associate_object_with_person(
        person_bbox: BBox,
        object_bbox: BBox,
        detection_confidence: float,
        class_name: str,
        person_track_id: int,
        temporal_score: float = 0.0,
        person_mask=None,
        object_mask=None,
        crop_padding: float = 0.20,
        source: str = "fullframe",
) -> Optional[ObjectCandidate]:
    """
    Attempt to associate one object detection with one person.

    Returns an ObjectCandidate if the spatial score is non-zero (i.e. the
    object is within range), otherwise None.

    The caller is responsible for feeding the returned candidate into
    ObjectTrackBuffer.update() to fill in temporal_score, and then
    calling calculate_evidence_score() to produce the final composite score.
    """
    spatial_score = calculate_spatial_score(person_bbox, object_bbox, crop_padding)
    if spatial_score <= 0.0:
        return None   # too far away, skip entirely

    seg_available = person_mask is not None and object_mask is not None
    seg_score = calculate_seg_score(person_mask, object_mask)

    cand = ObjectCandidate(
        class_name=class_name,
        confidence=detection_confidence,
        bbox=object_bbox,
        person_track_id=person_track_id,
        spatial_score=spatial_score,
        seg_score=seg_score,
        temporal_score=temporal_score,
        segmentation_available=seg_available,
        source=source,
    )
    return cand


# ── Temporal buffer ───────────────────────────────────────────────────────────
class _PairHistory:
    """History for one (person_track_id, class_name) pair."""
    __slots__ = ("hits", "conf_sum", "spatial_sum", "seg_sum")

    def __init__(self):
        self.hits = 0
        self.conf_sum = 0.0
        self.spatial_sum = 0.0
        self.seg_sum = 0.0

    def update(self, cand: ObjectCandidate):
        self.hits += 1
        self.conf_sum += cand.confidence
        self.spatial_sum += cand.spatial_score
        self.seg_sum += cand.seg_score

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / self.hits if self.hits else 0.0

    @property
    def mean_spatial(self) -> float:
        return self.spatial_sum / self.hits if self.hits else 0.0

    @property
    def mean_seg(self) -> float:
        return self.seg_sum / self.hits if self.hits else 0.0

    def temporal_score(self) -> float:
        """
        Score in [0, 1] based on how many frames this pair has been seen.

        hits=1  → 0.20  (just appeared)
        hits=2  → 0.50
        hits=3  → 0.75  (PROBABLE threshold crossed)
        hits=5  → 0.90
        hits≥8  → 1.00
        """
        return min(1.0, 0.15 + 0.85 * (1.0 - math.exp(-0.4 * (self.hits - 1))))


class ObjectTrackBuffer:
    """
    Per-camera temporal history of (person_track_id, class_name) pairs.

    Usage:
        buf = ObjectTrackBuffer()

        # per frame, after spatial association:
        confirmed, probable, candidates = buf.update(frame_candidates)
        # frame_candidates is a list of ObjectCandidate from associate_object_with_person()
    """

    def __init__(self,
                 min_confirm_frames: int = MIN_OBJECT_CONFIRM_FRAMES,
                 probable_frames: int = PROBABLE_FRAMES,
                 crop_padding: float = 0.20):
        self.min_confirm_frames = min_confirm_frames
        self.probable_frames = probable_frames
        self.crop_padding = crop_padding
        # key: (person_track_id, class_name) → _PairHistory
        self._history: dict[tuple, _PairHistory] = defaultdict(_PairHistory)

    def update(self, candidates: List[ObjectCandidate]) -> List[ObjectCandidate]:
        """
        Feed this frame's spatial candidates.
        Updates temporal scores and evidence scores in-place.
        Returns the same list with all scores and states populated.
        """
        for cand in candidates:
            key = (cand.person_track_id, cand.class_name)
            hist = self._history[key]
            hist.update(cand)

            cand.temporal_score = hist.temporal_score()
            cand.evidence_score = calculate_evidence_score(
                spatial_score=cand.spatial_score,
                temporal_score=cand.temporal_score,
                detection_confidence=cand.confidence,
                seg_score=cand.seg_score,
                segmentation_available=cand.segmentation_available,
            )
            cand.association_state = association_state_from_score(cand.evidence_score)

        return candidates

    def get_history(self, person_track_id: int, class_name: str) -> Optional[_PairHistory]:
        return self._history.get((person_track_id, class_name))
