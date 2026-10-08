"""
config.py — Central configuration for VISION-IQ Member 2 retrieval module.
All model IDs, paths, thresholds, weights, and camera->location map live here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)
CHROMA_DB_PATH = os.path.join(PROJECT_DIR, "data", "chroma")
EVENTS_DB_PATH = os.path.join(PROJECT_DIR, "data", "events_store.db")

FRAMES_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "frames")
CLIPS_DIR  = os.path.join(os.path.dirname(BASE_DIR), "data", "clips")
VIDEOS_DIR = os.path.join(os.path.dirname(BASE_DIR), "data", "videos")

# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME = "cctv_evidence"

# ---------------------------------------------------------------------------
# SigLIP2 model
# ---------------------------------------------------------------------------
SIGLIP_MODEL_ID = "google/siglip2-base-patch16-naflex"
EMBEDDING_DIM   = 768

# Text-query padding
TEXT_MAX_LENGTH = 64          # pad queries to 64 tokens (SigLIP2 spec)
TEXT_PADDING    = "max_length"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
SEMANTIC_TOP_K        = 50    # how many candidates to fetch from Chroma
RERANK_TOP_K          = 10    # how many to rerank
FINAL_TOP_K           = 3     # final results returned to caller

# ---------------------------------------------------------------------------
# Reranker weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
W_SEMANTIC  = 0.50
W_OBJECT    = 0.20
W_LOCATION  = 0.15
W_TEMPORAL  = 0.15

assert abs(W_SEMANTIC + W_OBJECT + W_LOCATION + W_TEMPORAL - 1.0) < 1e-6, \
    "Reranker weights must sum to 1.0"

# ---------------------------------------------------------------------------
# Confidence & no-match threshold
# ---------------------------------------------------------------------------
# Illustrative default — MUST be calibrated on your eval set.
# Raw SigLIP cosine scores are typically 0.05-0.30; the reranker uses
# normalized semantic scores, so the combined score is in [0, 1].
NO_MATCH_THRESHOLD = 0.50    # below this -> no_verified_match

# Attribute-probe margin threshold (Phase 3)
ATTRIBUTE_PROBE_MARGIN = 0.05  # min difference to emit an attribute reason

# ---------------------------------------------------------------------------
# Camera -> location map
# Add every camera that M1 supplies.
# ---------------------------------------------------------------------------
CAMERA_LOCATION_MAP: dict = {
    "G336": "main_gate",
    "G337": "parking_lot",
    "G338": "lobby",
    "G339": "side_entrance",
    "CAM_01": "main_gate",
    "CAM_02": "parking_lot",
    "CAM_03": "lobby",
    "CAM_04": "side_entrance",
    "cam_01": "main_gate",
    "cam_02": "parking_lot", 
    "cam_03": "lobby",
    "cam_04": "side_entrance",
}

# Helper to get location (case-insensitive)
def get_camera_location(camera_id: str) -> str:
    """Get location for camera ID (case-insensitive lookup)."""
    return CAMERA_LOCATION_MAP.get(camera_id) or CAMERA_LOCATION_MAP.get(camera_id.lower(), "unknown_location")

# Shorthand
CAMERA_LOCATIONS = CAMERA_LOCATION_MAP  # Alias for backward compatibility

# Reverse map: location -> list of camera_ids
LOCATION_CAMERA_MAP: dict = {}
for _cam, _loc in CAMERA_LOCATION_MAP.items():
    LOCATION_CAMERA_MAP.setdefault(_loc, []).append(_cam)

# ---------------------------------------------------------------------------
# Temporal query defaults
# ---------------------------------------------------------------------------
TEMPORAL_DECAY_RATE = 0.0001   # score = exp(-decay * seconds_outside_window)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
TIMESTAMP_TOLERANCE_SEC = 5.0   # correct if |predicted - gt| <= this
