"""
events/event_search.py
Structured event retrieval interface for Member 3 (Backend / LLM / CLI).
Searches through CVEvent metadata using structured and metadata filters.
Generates ZERO embeddings.
"""

from __future__ import annotations

import re
import sys
import os
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from events.event_store import count_events, get_all_events

# Stopwords that do not carry semantic/filtering meaning in surveillance queries
_STOPWORDS = {
    "show", "find", "get", "list", "search", "events", "event",
    "from", "in", "at", "the", "a", "an", "of", "for", "with",
    "near", "around", "on", "to", "who", "what", "which", "is",
    "are", "was", "were", "been", "there", "did", "any"
}

_BAG_TERMS = {"handbag", "purse", "bag", "backpack", "suitcase"}
_PERSON_TERMS = {"person", "people", "man", "woman", "pedestrian"}
_VEHICLE_TERMS = {"car", "vehicle", "truck", "bus", "motorcycle", "bike"}


def _parse_query_constraints(query: str) -> Dict[str, Any]:
    """Parse structured constraints from query text."""
    q = query.strip().lower()
    constraints: Dict[str, Any] = {
        "camera_ids": [],
        "object_terms": set(),
        "timestamp_target": None,
        "is_roi_alert": False,
        "is_suspicious_alert": False,
        "relationship": None,
        "location": None,
        "keywords": set(),
    }

    # Camera pattern: cam_01, cam_03, cam 03, camera 3, g336, etc.
    cam_matches = re.findall(r"\b(?:cam(?:era)?[\s_-]*0?(\d+)|([a-z]\d{3,4}))\b", q)
    for num, code in cam_matches:
        if num:
            constraints["camera_ids"].extend([f"cam_{int(num):02d}", f"cam_{int(num)}", f"CAM_{int(num):02d}"])
        if code:
            constraints["camera_ids"].append(code.upper())
            constraints["camera_ids"].append(code.lower())

    # Check against known camera location mapping
    for cam_name in getattr(config, "CAMERA_LOCATION_MAP", {}):
        if re.search(r"\b" + re.escape(cam_name.lower()) + r"\b", q):
            constraints["camera_ids"].append(cam_name)

    # Location matching
    for loc_key in getattr(config, "LOCATION_CAMERA_MAP", {}):
        if loc_key.replace("_", " ") in q or loc_key in q:
            constraints["location"] = loc_key
            break

    # Object classes
    tokens = set(re.findall(r"\b[a-z0-9_-]+\b", q))
    if tokens & _BAG_TERMS:
        constraints["object_terms"].update(tokens & _BAG_TERMS)
    if tokens & _PERSON_TERMS:
        constraints["object_terms"].update(tokens & _PERSON_TERMS)
    if tokens & _VEHICLE_TERMS:
        constraints["object_terms"].update(tokens & _VEHICLE_TERMS)

    # Alerts
    if any(k in q for k in ["danger zone", "danger", "roi", "restricted", "intrusion"]):
        constraints["is_roi_alert"] = True
    if any(k in q for k in ["suspicious-object", "suspicious object", "suspicious", "abandoned", "alert"]):
        constraints["is_suspicious_alert"] = True

    # Relationship / Actions
    if any(k in q for k in ["carrying", "carry", "carries", "holding", "hold"]):
        constraints["relationship"] = "carrying"

    # Timestamp mentions (e.g. "around 3 seconds", "at 3.45 seconds", "around 3s", "3.46")
    ts_match = re.search(r"\b(?:around|at|near|time|timestamp)?\s*(\d+(?:\.\d+)?)\s*(?:seconds?|sec|s\b)", q)
    if ts_match:
        try:
            constraints["timestamp_target"] = float(ts_match.group(1))
        except ValueError:
            pass

    # Generic keywords
    keywords = {t for t in tokens if t not in _STOPWORDS and len(t) > 2}
    constraints["keywords"] = keywords

    return constraints


def _score_event(event: Dict[str, Any], constraints: Dict[str, Any], explicit_filters: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Score event relevance against constraints.
    Returns (score, reasons).
    """
    score = 0.0
    matched_reasons = []

    cam_id = str(event.get("camera_id") or "").lower()
    location = str(event.get("location") or "").lower()
    obj_class = str(event.get("object_class") or "").lower()
    rel = str(event.get("relationship") or "").lower()
    desc = str(event.get("description") or "").lower()
    search_text = str(event.get("search_text") or "").lower()
    tags = [str(t).lower() for t in event.get("semantic_tags") or []]
    timestamp_sec = float(event.get("timestamp_sec") or 0.0)

    # 1. Explicit caller filters
    if explicit_filters.get("camera_id"):
        if cam_id != str(explicit_filters["camera_id"]).lower():
            return 0.0, []
        score += 2.0
        matched_reasons.append(f"camera {cam_id}")

    if explicit_filters.get("location"):
        if location != str(explicit_filters["location"]).lower():
            return 0.0, []
        score += 2.0
        matched_reasons.append(f"location {location}")

    if explicit_filters.get("relationship"):
        if rel != str(explicit_filters["relationship"]).lower():
            return 0.0, []
        score += 2.0
        matched_reasons.append(f"relationship {rel}")

    # 2. Inferred camera constraints
    if constraints["camera_ids"]:
        cam_match = any(c.lower() == cam_id or c.lower() in cam_id for c in constraints["camera_ids"])
        if not cam_match:
            return 0.0, []
        score += 3.0
        matched_reasons.append(f"camera {cam_id}")

    # 3. Location constraints
    if constraints["location"]:
        if location == constraints["location"].lower():
            score += 2.0
            matched_reasons.append(f"location {location}")
        else:
            return 0.0, []

    # 4. Object terms
    if constraints["object_terms"]:
        obj_matched = False
        if constraints["object_terms"] & _BAG_TERMS:
            if event.get("has_bag") or obj_class in _BAG_TERMS or any(t in _BAG_TERMS for t in tags):
                score += 2.5
                obj_matched = True
                matched_reasons.append(f"object {obj_class or 'bag'}")
        if constraints["object_terms"] & _PERSON_TERMS:
            if event.get("has_person") or obj_class == "person" or any(t in _PERSON_TERMS for t in tags):
                score += 2.0
                obj_matched = True
                matched_reasons.append("person")
        if not obj_matched:
            return 0.0, []

    # 5. Timestamp constraint
    if constraints["timestamp_target"] is not None:
        target = constraints["timestamp_target"]
        diff = abs(timestamp_sec - target)
        if diff <= 5.0:
            proximity_score = max(0.0, 3.0 * (1.0 - (diff / 5.0)))
            score += proximity_score
            matched_reasons.append(f"timestamp {timestamp_sec:.2f}s (near {target}s)")
        else:
            return 0.0, []

    # 6. ROI Alert
    if constraints["is_roi_alert"]:
        if event.get("roi_alert") or "roi" in tags or "danger" in search_text:
            score += 3.0
            matched_reasons.append("ROI alert / danger zone")
        else:
            return 0.0, []

    # 7. Suspicious Alert
    if constraints["is_suspicious_alert"]:
        if event.get("suspicious_object_alert") or "alert" in tags or "suspicious" in search_text:
            score += 3.0
            matched_reasons.append("suspicious alert")
        else:
            return 0.0, []

    # 8. Relationship
    if constraints["relationship"]:
        if rel == constraints["relationship"] or constraints["relationship"] in search_text or constraints["relationship"] in desc:
            score += 2.0
            matched_reasons.append(f"action {constraints['relationship']}")
        else:
            return 0.0, []

    # 9. Keyword matching on search_text, tags, description
    full_text = f"{desc} {search_text} {' '.join(tags)}"
    matched_kws = 0
    for kw in constraints["keywords"]:
        if kw in full_text:
            matched_kws += 1
            score += 1.0

    if not matched_reasons and matched_kws == 0:
        return 0.0, []

    return score, matched_reasons


def search_events(
    query: str,
    top_k: int = 5,
    camera_id: Optional[str] = None,
    location: Optional[str] = None,
    relationship: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search CV events using structured metadata filtering.
    ZERO embeddings are generated.
    """
    total = count_events()
    if total == 0:
        return {
            "status": "no_events",
            "query": query,
            "total_events_in_db": 0,
            "results_count": 0,
            "events": [],
            "message": "No CV events currently indexed in structured storage."
        }

    constraints = _parse_query_constraints(query)
    explicit_filters = {
        "camera_id": camera_id,
        "location": location,
        "relationship": relationship,
    }

    all_events = get_all_events()
    candidates: List[Tuple[float, Dict[str, Any]]] = []

    for event in all_events:
        score, reasons = _score_event(event, constraints, explicit_filters)
        if score > 0.0:
            candidates.append((score, event))

    # Sort descending by score, tie-break by timestamp
    candidates.sort(key=lambda item: (item[0], item[1].get("timestamp_sec", 0.0)), reverse=True)

    max_score = candidates[0][0] if candidates else 1.0
    results: List[Dict[str, Any]] = []

    for rank, (score, evt) in enumerate(candidates[:top_k], start=1):
        norm_similarity = round(min(1.0, max(0.50, score / max_score if max_score > 0 else 1.0)), 3)
        results.append({
            "rank": rank,
            "event_id": evt["event_id"],
            "similarity": norm_similarity,
            "description": evt["description"],
            "camera_id": evt["camera_id"],
            "location": evt["location"],
            "time_start": evt["time_start"],
            "time_end": evt["time_end"],
            "timestamp": evt.get("timestamp_sec"),
            "relationship": evt["relationship"],
            "event_type": evt.get("raw_event", {}).get("event_type", ""),
            "evidence": {
                "frame_path": evt["frame_path"],
                "crop_path": evt["crop_path"],
                "clip_path": evt["clip_path"],
            }
        })

    return {
        "status": "success",
        "query": query,
        "total_events_in_db": total,
        "results_count": len(results),
        "events": results
    }
