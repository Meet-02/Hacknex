"""
retrieval/search.py
Public search() function — the only interface M3 needs to call.
M3 does not need to know about SigLIP2 or ChromaDB.

Usage:
    from retrieval.search import search
    result = search("Did a red car enter the main gate?")
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Dict, Any
import config
from retrieval.query_parser  import parse_query
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker      import rerank
from rag.context_builder     import build_context


def search(
    query: str,
    top_k: int = config.FINAL_TOP_K,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    query   : Natural-language query.
    top_k   : Max evidence items to return.
    filters : Optional caller-supplied overrides:
              {"camera_id": "G336"} or {"location": "main_gate"}
              or {"time_start": 50400.0, "time_end": 54000.0}.

    Returns
    -------
    Unified evidence envelope (see Section 7 of workflow doc).
    """
    parsed = parse_query(query)

    # Apply explicit caller overrides
    if filters:
        if "camera_id" in filters:
            cam = filters["camera_id"]
            parsed["camera_ids"] = [cam]
            parsed["location"]   = config.CAMERA_LOCATION_MAP.get(cam)
        if "location" in filters:
            loc = filters["location"]
            parsed["location"]   = loc
            parsed["camera_ids"] = config.LOCATION_CAMERA_MAP.get(loc, [])
        if "time_start" in filters:
            parsed["time_start"] = float(filters["time_start"])
        if "time_end" in filters:
            parsed["time_end"] = float(filters["time_end"])

    hits, filter_relaxed = hybrid_search(query, parsed=parsed)
    ranked   = rerank(hits, parsed, top_k=config.RERANK_TOP_K)
    envelope = build_context(
        query_text=query,
        ranked_hits=ranked,
        parsed=parsed,
        filter_relaxed=filter_relaxed,
        top_k=top_k,
    )
    return envelope
