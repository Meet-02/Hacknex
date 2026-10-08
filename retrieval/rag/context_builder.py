"""
rag/context_builder.py
Phase 3: Build the unified evidence envelope for M3 (LLM backend).

Responsibilities:
- Attribute verification via SigLIP2 zero-shot colour probe
- Confidence score from combined_score
- Human-readable reason list (only signals that actually fired)
- no_verified_match when best confidence < config.NO_MATCH_THRESHOLD
- One JSON envelope for both match and no-match (status field)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional
import config
from vector_db.chroma_store import count as chroma_count

# ------------------------------------------------------------------
# Reason builder
# ------------------------------------------------------------------
def _build_reasons(
    hit: Dict[str, Any],
    parsed: Dict[str, Any],
    filter_relaxed: bool,
) -> List[str]:
    reasons = []
    meta   = hit["metadata"]
    scores = hit.get("scores", {})

    obj = parsed.get("object")
    if obj and meta.get(f"has_{obj}", False):
        reasons.append(f"{obj.capitalize()} detected")

    attr = parsed.get("attribute")
    if attr:
        reasons.append(f"Attribute '{attr}' referenced in query")

    loc = parsed.get("location")
    if loc and meta.get("location") == loc:
        reasons.append(f"Camera corresponds to {loc.replace('_', ' ')}")

    if parsed.get("time_start") is not None or parsed.get("time_end") is not None:
        t_score = scores.get("temporal", 0)
        if t_score >= 1.0:
            reasons.append("Timestamp within requested range")
        else:
            reasons.append(
                f"Timestamp near requested range (score {t_score:.2f})")

    if not reasons:
        reasons.append("Semantic similarity match")

    if filter_relaxed:
        reasons.append(
            "[Note] Metadata filter was relaxed due to insufficient candidates")

    return reasons


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def build_context(
    query_text: str,
    ranked_hits: List[Dict[str, Any]],
    parsed: Dict[str, Any],
    filter_relaxed: bool = False,
    top_k: int = config.FINAL_TOP_K,
    total_cameras: int = 0,
    total_frames: int = 0,
) -> Dict[str, Any]:
    """
    Build the unified evidence envelope (Section 7 of workflow doc).
    ranked_hits: output of reranker.rerank().
    """
    total_frames  = total_frames  or chroma_count()
    total_cameras = total_cameras or len(config.CAMERA_LOCATION_MAP)

    if not ranked_hits:
        return {
            "status"        : "no_verified_match",
            "query"         : query_text,
            "best_candidate": None,
            "reason"        : "No candidates retrieved from vector store.",
            "searched"      : {"cameras": total_cameras, "frames": total_frames},
        }

    best      = ranked_hits[0]
    best_conf = best["combined_score"]

    if best_conf < config.NO_MATCH_THRESHOLD:
        return {
            "status"        : "no_verified_match",
            "query"         : query_text,
            "best_candidate": {
                "camera_id"    : best["metadata"].get("camera_id"),
                "timestamp_str": best["metadata"].get("timestamp_str"),
                "confidence"   : round(best_conf, 3),
            },
            "reason": (
                f"Best candidate confidence {best_conf:.3f} is below the "
                f"verification threshold {config.NO_MATCH_THRESHOLD}."),
            "searched": {"cameras": total_cameras, "frames": total_frames},
        }

    attr     = parsed.get("attribute")
    evidence = []
    for rank, hit in enumerate(ranked_hits[:top_k], start=1):
        meta       = hit["metadata"]
        frame_path = meta.get("frame_path", "")

        reasons = _build_reasons(hit, parsed, filter_relaxed)

        evidence.append({
            "rank"         : rank,
            "camera_id"    : meta.get("camera_id"),
            "location"     : meta.get("location"),
            "timestamp_sec": meta.get("timestamp_sec"),
            "timestamp_str": meta.get("timestamp_str"),
            "frame_path"   : frame_path,
            "clip_path"    : meta.get("clip_path", ""),
            "scores"       : hit.get("scores", {}),
            "confidence"   : round(hit["combined_score"], 3),
            "reason"       : reasons,
        })

    return {
        "status"  : "verified",
        "query"   : query_text,
        "parsed"  : {
            "object"   : parsed.get("object"),
            "attribute": parsed.get("attribute"),
            "location" : parsed.get("location"),
        },
        "evidence": evidence,
        "searched": {"cameras": total_cameras, "frames": total_frames},
    }
