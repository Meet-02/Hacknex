"""
retrieval/query_parser.py
Rule-based NL query parser (Phase 2).
Extracts: object, attribute, location, time_start, time_end, event.
M3 can replace this with an LLM planner; the dict interface stays the same.
"""

import re
from typing import Optional, Dict, Any, List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ----------------------------------------------------------------
# Keyword sets
# ----------------------------------------------------------------
OBJECT_KEYWORDS = {
    "car", "vehicle", "truck", "bus", "motorcycle", "motorbike",
    "bicycle", "bike", "person", "people", "man", "woman",
    "child", "pedestrian",
}

COLOR_KEYWORDS = {
    "red", "blue", "green", "yellow", "white", "black", "silver",
    "grey", "gray", "orange", "brown", "purple",
}

LOCATION_KEYWORDS: set = set(config.LOCATION_CAMERA_MAP.keys())

EVENT_KEYWORDS = {
    "enter", "entered", "exit", "exited", "leave", "left",
    "walk", "walked", "run", "ran", "park", "parked",
    "carry", "carrying", "hold", "holding",
}

# Time-of-day patterns (24h and 12h)
_TIME_RE = re.compile(
    r"(?:after|before|at|around|from|between)?\s*"
    r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"
    r"(?:\s*(?:to|-|and)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?",
    re.IGNORECASE,
)


def _to_seconds(h: str, m: Optional[str], ampm: Optional[str]) -> float:
    hour   = int(h)
    minute = int(m) if m else 0
    if ampm:
        if ampm.lower() == "pm" and hour != 12:
            hour += 12
        elif ampm.lower() == "am" and hour == 12:
            hour = 0
    return float(hour * 3600 + minute * 60)


def parse_query(query: str) -> Dict[str, Any]:
    """
    Returns:
        {
            "object"    : str | None,
            "attribute" : str | None,   # colour / appearance
            "location"  : str | None,
            "time_start": float | None, # seconds since midnight
            "time_end"  : float | None,
            "event"     : str | None,
            "camera_ids": list[str],    # cameras matching location
        }
    """
    q      = query.lower()
    tokens = re.findall(r"\b\w+\b", q)

    # Object
    obj = next((t for t in tokens if t in OBJECT_KEYWORDS), None)
    # Synonym normalisation
    if obj in {"people", "man", "woman", "child", "pedestrian"}:
        obj = "person"
    if obj in {"motorbike"}:
        obj = "motorcycle"

    # Attribute (colour)
    attr = next((t for t in tokens if t in COLOR_KEYWORDS), None)

    # Location — single-word then multi-word fallback
    loc = next((t for t in tokens if t in LOCATION_KEYWORDS), None)
    if loc is None:
        for loc_key in LOCATION_KEYWORDS:
            if loc_key.replace("_", " ") in q:
                loc = loc_key
                break

    cam_ids: List[str] = config.LOCATION_CAMERA_MAP.get(loc, []) if loc else []

    # Event
    event = next((t for t in tokens if t in EVENT_KEYWORDS), None)

    # Time
    time_start: Optional[float] = None
    time_end:   Optional[float] = None
    tm = _TIME_RE.search(q)
    if tm:
        h1, m1, ap1, h2, m2, ap2 = tm.groups()
        if h1:
            time_start = _to_seconds(h1, m1, ap1)
        if h2:
            time_end = _to_seconds(h2, m2, ap2)

    return {
        "object"    : obj,
        "attribute" : attr,
        "location"  : loc,
        "time_start": time_start,
        "time_end"  : time_end,
        "event"     : event,
        "camera_ids": cam_ids,
    }
