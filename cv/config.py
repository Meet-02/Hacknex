"""
config.py

Central configuration for the Hacknex CV pipeline.
All tunable values live here so that no file hard-codes paths or thresholds.

Class categories
────────────────
PERSON_CLASSES           - people
VEHICLE_CLASSES          - motorised / pedal vehicles (tracked separately)
PERSON_ASSOCIABLE_CLASSES- objects a person could plausibly carry/hold/wear.
                           Raw detections of these are *candidates*; they still
                           need to pass spatial association + temporal
                           confirmation before being reported as confirmed.
RELEVANT_CLASSES         - union of the above (everything the pipeline cares about)

Background / environment classes (chair, TV, bed, dining table, toilet, …) are
intentionally excluded; they are irrelevant for person-object association and
their presence would only generate noise.

Thresholds
──────────
Every threshold has a comment explaining what it controls and what happens if
you raise or lower it.

Model paths
───────────
DETECTION_MODEL    - detection weights (yolo26s.pt preferred; yolo26n.pt fallback)
SEGMENTATION_MODEL - segmentation weights (optional; pipeline continues without it)
"""

import os

# ── Model paths ───────────────────────────────────────────────────────────────
# The pipeline looks for these relative to this file's directory (cv/).
# Place yolo26n-seg.pt in cv/ to enable segmentation.
_HERE = os.path.dirname(os.path.abspath(__file__))

DETECTION_MODEL    = os.path.join(_HERE, "yolo26s.pt")   # preferred (more accurate)
_DET_FALLBACK      = os.path.join(_HERE, "yolo26n.pt")   # auto-fallback if s not found
SEGMENTATION_MODEL = os.path.join(_HERE, "yolo26n-seg.pt")  # optional

if not os.path.exists(DETECTION_MODEL):
    DETECTION_MODEL = _DET_FALLBACK

# ── Authoritative camera source mapping ───────────────────────────────────────
CAMERA_SOURCES: dict[str, str] = {
    "cam_01": os.path.join(_HERE, "videos", "cam_01.mp4"),
    "cam_02": os.path.join(_HERE, "videos", "cam_02.mp4"),
    "cam_03": os.path.join(_HERE, "videos", "cam_03.mp4"),
}
CAMERAS = CAMERA_SOURCES  # alias for backward compatibility

# ── Detection thresholds ──────────────────────────────────────────────────────
# Stage A: full-frame detection
PERSON_CONF  = 0.45   # minimum confidence to accept a person detection
OBJECT_CONF  = 0.20   # minimum confidence for object candidates in person crops (Stage B)
                       # intentionally lower than PERSON_CONF because small objects
                       # score lower; false positives are filtered later by association

# Stage B: person-crop second-pass detection
# Run YOLO on the padded person crop at this confidence to catch small objects
CROP_CONF    = 0.20   # (alias; same value as OBJECT_CONF for clarity in pipeline)

# ── Person crop ───────────────────────────────────────────────────────────────
PERSON_CROP_PADDING = 0.20   # expand person bbox by this fraction on each side
                              # captures hands / bags that stick outside the bbox

# ── Class categories ──────────────────────────────────────────────────────────
PERSON_CLASSES = {"person"}

VEHICLE_CLASSES = {
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}

# All COCO classes that a person could plausibly carry, hold, or wear.
# These are CANDIDATES, not automatic associations — confidence, spatial
# proximity, and temporal consistency must all support the association.
PERSON_ASSOCIABLE_CLASSES = {
    # bags / luggage
    "backpack",
    "handbag",
    "suitcase",
    # clothing accessories
    "umbrella",
    "tie",
    # hand-held items
    "bottle",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "laptop",
    "cell phone",
    "book",
    "scissors",
    "vase",
    "toothbrush",
    "hair drier",
    "teddy bear",
    # sports equipment (held / carried)
    "sports ball",
    "frisbee",
    "skis",
    "snowboard",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "kite",
}

# Everything the pipeline cares about (union)
RELEVANT_CLASSES = PERSON_CLASSES | VEHICLE_CLASSES | PERSON_ASSOCIABLE_CLASSES

# Kept for backward compatibility with code that still imports CARRIED_OBJECT_CLASSES
CARRIED_OBJECT_CLASSES = PERSON_ASSOCIABLE_CLASSES


def category_of(class_name: str) -> str:
    """Return 'person' | 'vehicle' | 'carried_object' | 'other'."""
    if class_name in PERSON_CLASSES:
        return "person"
    if class_name in VEHICLE_CLASSES:
        return "vehicle"
    if class_name in PERSON_ASSOCIABLE_CLASSES:
        return "carried_object"
    return "other"


# ── Tracker / temporal confirmation ───────────────────────────────────────────
MIN_PERSON_TRACK_FRAMES  = 3   # processed frames before a person is "reliable"
                                # lower → faster to trust new people; raises false-positive risk

MIN_OBJECT_CONFIRM_FRAMES = 3  # same class must appear associated with same person
                                # in this many frames before association is "confirmed"

PROBABLE_FRAMES = 2            # frames needed to reach "probable" (intermediate state)

# ── Association thresholds ────────────────────────────────────────────────────
# Evidence score is composed of:
#   spatial_score  (0–1) based on proximity / overlap with person bbox
#   temporal_score (0–1) based on how many frames the pairing has been seen
#   seg_score      (0–1) mask overlap if segmentation is available
#   confidence     raw YOLO detection confidence

SPATIAL_WEIGHT    = 0.35
TEMPORAL_WEIGHT   = 0.40
SEG_WEIGHT        = 0.15
CONFIDENCE_WEIGHT = 0.10

# Thresholds for association state promotion
CONFIRMED_ASSOCIATION_THRESHOLD = 0.70   # evidence_score >= this → "confirmed"
PROBABLE_ASSOCIATION_THRESHOLD  = 0.45   # evidence_score >= this → "probable"
                                          # below this → "candidate"

# Maximum distance between object centre and person bbox edge,
# expressed as a fraction of the person's larger dimension (width or height).
# Objects further than this are not associated at all.
MAX_ASSOC_DISTANCE_FACTOR = 1.0

# ── Evidence / output ─────────────────────────────────────────────────────────
EVIDENCE_SAVE_MIN_STATE = "probable"  # save evidence frames for this state and above
                                       # "candidate" | "probable" | "confirmed"

# ── Suspicious-object alerting ────────────────────────────────────────────────
# Each entry: class_name → {severity, alert_label}
#   severity options: "low" | "medium" | "high" | "critical"
#
# "gun" is NOT a COCO class; keeping it here allows future custom-model support.
# The pipeline will warn at startup for any class the loaded model does not support.
# Raw detections still need temporal confirmation + evidence threshold before an
# alert is created — one weak detection does NOT trigger an alert.
SUSPICIOUS_OBJECTS: dict = {
    "knife": {
        "severity":    "high",
        "alert_label": "Potential knife detected",
    },
    "scissors": {
        "severity":    "medium",
        "alert_label": "Potential scissors detected",
    },
    "gun": {
        "severity":    "critical",
        "alert_label": "Potential firearm detected",
    },
}

# Minimum YOLO detection confidence for a suspicious object to be accepted at all.
# This is applied BEFORE the temporal/evidence pipeline.
ALERT_MIN_CONF = 0.50

# Minimum evidence score for a suspicious-object association to become an alert.
# Reuses the same evidence scoring as normal objects.
ALERT_MIN_EVIDENCE_SCORE = 0.70

# Object synonyms used when building search_text / semantic_tags.
# Keys must be COCO class names; values are additional terms (do not add impossible facts).
OBJECT_SYNONYMS: dict = {
    "handbag":  ["purse", "bag"],
    "backpack": ["bag", "carried bag", "rucksack"],
    "suitcase": ["luggage", "baggage", "trolley"],
    "knife":    ["potential weapon", "suspicious object"],
    "scissors": ["potential sharp object", "suspicious object"],
    "gun":      ["potential firearm", "potential weapon", "suspicious object"],
}
