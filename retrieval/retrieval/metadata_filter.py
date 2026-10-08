"""
retrieval/metadata_filter.py
Builds Chroma where-clause dicts from parsed query constraints.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, Optional, List


def build_filter(parsed: Dict[str, Any]) -> Optional[Dict]:
    """
    Build a Chroma where-clause from parsed query fields.
    Returns None if no filtering constraints exist.

    Chroma filter syntax:
        {"$and": [{"field": {"$eq": val}}, ...]}
        {"$or":  [{"camera_id": {"$eq": "G336"}}, ...]}
    """
    conditions: List[Dict] = []

    # Camera / location
    cam_ids: List[str] = parsed.get("camera_ids", [])
    if cam_ids:
        if len(cam_ids) == 1:
            conditions.append({"camera_id": {"$eq": cam_ids[0]}})
        else:
            conditions.append({"$or": [{"camera_id": {"$eq": c}} for c in cam_ids]})

    # Object class via boolean flag
    obj = parsed.get("object")
    if obj:
        conditions.append({f"has_{obj}": {"$eq": True}})

    # Temporal filter
    time_start = parsed.get("time_start")
    time_end   = parsed.get("time_end")
    if time_start is not None:
        conditions.append({"timestamp_sec": {"$gte": float(time_start)}})
    if time_end is not None:
        conditions.append({"timestamp_sec": {"$lte": float(time_end)}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def relax_filter(where: Optional[Dict]) -> Optional[Dict]:
    """
    Return a less restrictive filter by dropping object-class conditions.
    Called when the strict filter returns too few candidates.
    """
    if where is None:
        return None
    if "$and" in where:
        relaxed = [c for c in where["$and"] if not _is_object_condition(c)]
        if not relaxed:
            return None
        if len(relaxed) == 1:
            return relaxed[0]
        return {"$and": relaxed}
    if _is_object_condition(where):
        return None
    return where


def _is_object_condition(cond: Dict) -> bool:
    return any(k.startswith("has_") for k in cond)
