"""
retrieval/hybrid_search.py
Metadata filtering + semantic search with auto-fallback when the
strict filter returns too few candidates.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional, Tuple
import config
from retrieval.semantic_search  import semantic_search
from retrieval.metadata_filter  import build_filter, relax_filter
from retrieval.query_parser     import parse_query

MIN_CANDIDATES = 5   # if fewer after strict filter, relax and retry


def hybrid_search(
    query_text: str,
    parsed: Optional[Dict[str, Any]] = None,
    top_k: int = config.SEMANTIC_TOP_K,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Returns (hits, filter_relaxed).
    filter_relaxed=True is surfaced in the reason list.
    """
    if parsed is None:
        parsed = parse_query(query_text)

    where          = build_filter(parsed)
    hits           = semantic_search(query_text, top_k=top_k, where=where)
    filter_relaxed = False

    if where is not None and len(hits) < MIN_CANDIDATES:
        relaxed_where = relax_filter(where)
        hits_relaxed  = semantic_search(query_text, top_k=top_k, where=relaxed_where)
        if len(hits_relaxed) > len(hits):
            hits           = hits_relaxed
            filter_relaxed = True

    return hits, filter_relaxed
