"""
[DEPRECATED / UNUSED IN ACTIVE PIPELINE]
retrieval/semantic_search.py
Old text embedding semantic search.
Disconnected from old SigLIP embedding generation.
Zero embeddings are generated.
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from events.event_search import search_events


def semantic_search(
    query_text: str,
    top_k: int = config.SEMANTIC_TOP_K,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Structured retrieval fallback (zero embeddings).
    Returns list of hit dicts: {id, metadata, similarity}.
    """
    res = search_events(query_text, top_k=top_k)
    hits = []
    for evt in res.get("events", []):
        hits.append({
            "id": evt.get("event_id"),
            "metadata": {
                "camera_id": evt.get("camera_id"),
                "location": evt.get("location"),
                "time_start": evt.get("time_start"),
                "time_end": evt.get("time_end"),
                "timestamp_sec": evt.get("timestamp"),
                "relationship": evt.get("relationship"),
                "frame_path": evt.get("evidence", {}).get("frame_path"),
                "crop_path": evt.get("evidence", {}).get("crop_path"),
                "clip_path": evt.get("evidence", {}).get("clip_path"),
            },
            "similarity": evt.get("similarity", 1.0),
        })
    return hits
