"""
retrieval/reranker.py
Reranks candidates using a weighted combination of four signals:

    final = 0.50 * semantic_norm
          + 0.20 * object_match
          + 0.15 * location_match
          + 0.15 * temporal_match

All components are in [0, 1].
semantic_norm: min-max normalised within the candidate set so raw SigLIP
cosine magnitudes (typically 0.05-0.30) do not skew the combined score.
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional
import config


def _normalise(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _temporal_match(
    ts_sec: float,
    time_start: Optional[float],
    time_end:   Optional[float],
) -> float:
    """1.0 inside window; decays exponentially outside."""
    if time_start is None and time_end is None:
        return 1.0
    inside, distance = True, 0.0
    if time_start is not None and ts_sec < time_start:
        inside, distance = False, time_start - ts_sec
    if time_end is not None and ts_sec > time_end:
        inside = False
        distance = max(distance, ts_sec - time_end)
    return 1.0 if inside else math.exp(-config.TEMPORAL_DECAY_RATE * distance)


def rerank(
    hits: List[Dict[str, Any]],
    parsed: Dict[str, Any],
    top_k: int = config.RERANK_TOP_K,
) -> List[Dict[str, Any]]:
    """
    hits: list of {id, metadata, similarity} from semantic_search.
    parsed: output of parse_query.
    Returns top_k dicts enriched with: scores, combined_score.
    """
    if not hits:
        return []

    candidates = hits[: max(top_k * 3, config.SEMANTIC_TOP_K)]
    raw_sem    = [h["similarity"] for h in candidates]
    sem_norm   = _normalise(raw_sem)

    obj_req  = parsed.get("object")
    loc_req  = parsed.get("location")
    t_start  = parsed.get("time_start")
    t_end    = parsed.get("time_end")

    scored = []
    for h, s_n in zip(candidates, sem_norm):
        meta = h["metadata"]

        # Object match via M1 boolean flag
        if obj_req:
            obj_score = 1.0 if meta.get(f"has_{obj_req}", False) else 0.0
        else:
            obj_score = 1.0   # no constraint -> full score

        # Location match
        loc_score = (
            1.0 if (not loc_req or meta.get("location") == loc_req) else 0.0
        )

        # Temporal match
        ts_sec    = float(meta.get("timestamp_sec", 0.0))
        temp_score = _temporal_match(ts_sec, t_start, t_end)

        combined = (
            config.W_SEMANTIC  * s_n        +
            config.W_OBJECT    * obj_score  +
            config.W_LOCATION  * loc_score  +
            config.W_TEMPORAL  * temp_score
        )

        scored.append({
            **h,
            "scores": {
                "semantic" : round(s_n,         4),
                "object"   : round(obj_score,   4),
                "location" : round(loc_score,   4),
                "temporal" : round(temp_score,  4),
            },
            "combined_score": round(combined, 4),
        })

    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    return scored[:top_k]
