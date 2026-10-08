"""
events/index_events.py
Ingests Member 1 CVEvent JSON objects, validates, serializes to canonical text,
and stores in structured event storage without generating embeddings.

Usage:
    from events.index_events import ingest_event, ingest_events_list
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from events.canonical_serializer import serialize_event_to_text
from events.event_store import upsert_events
from events.ingest_detections import convert_detection_to_event


def _event_to_record(event: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a CVEvent dict into a structured storage record (zero embeddings)."""
    if "camera_id" in event:
        event = convert_detection_to_event(event)
    evt_id = event["event_id"]
    description = serialize_event_to_text(event)

    cam_info = event.get("camera", {})
    time_info = event.get("time", {})
    entities = event.get("entities", {})
    person = entities.get("person", {})
    obj = entities.get("object", {})
    rel = event.get("relationship", {})
    evidence = event.get("evidence", {})

    metadata = {
        "event_id": evt_id,
        "event_type": str(event.get("event_type", "")),
        "camera_id": str(cam_info.get("camera_id", "")),
        "location": str(cam_info.get("location", "")),
        "time_start": str(time_info.get("start", "")),
        "time_end": str(time_info.get("end", "")),
        "person_track_id": int(person.get("track_id") if person.get("track_id") is not None else -1) if person else -1,
        "object_class": str(obj.get("class", "")) if obj else "",
        "relationship": str(rel.get("type", "")) if rel else "",
        "relationship_conf": float(rel.get("confidence", 0.0)) if rel else 0.0,
        "frame_path": str(evidence.get("frame_path") or ""),
        "clip_path": str(evidence.get("clip_path") or ""),
        "crop_path": str(evidence.get("crop_path") or ""),
        "timestamp_sec": float(time_info.get("timestamp_sec", 0.0)),
        "object_confidence": float(obj.get("confidence", 0.0) or 0.0),
        "association_state": str(obj.get("association_state", "")),
        "has_person": bool(person),
        "has_bag": str(obj.get("class", "")) in {"handbag", "backpack", "suitcase", "bag", "purse"},
        "roi_alert": bool(event.get("roi_alert", False)),
        "suspicious_object_alert": bool(event.get("suspicious_object_alert", False)),
        "description": str(event.get("description", description)),
    }

    return {
        "id": evt_id,
        "metadata": metadata,
        "document": description,
        "raw_event": event,
    }


# Backward compatibility alias
_event_to_chroma_record = _event_to_record


def ingest_event(event: Dict[str, Any]) -> str:
    """Ingest a single CVEvent JSON object with ZERO embeddings."""
    rec = _event_to_record(event)
    upsert_events([rec])
    return rec["id"]


def ingest_events_list(events: List[Dict[str, Any]]) -> int:
    """Ingest a list of CVEvent JSON objects with ZERO embeddings."""
    records = []
    for evt in events:
        records.append(_event_to_record(evt))
    if records:
        upsert_events(records)
    return len(records)


def ingest_events_from_file(json_file_path: str) -> int:
    """Ingest events from a .json or .jsonl file with ZERO embeddings."""
    events = []
    with open(json_file_path, "r", encoding="utf-8") as f:
        if json_file_path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        else:
            data = json.load(f)
            events = data if isinstance(data, list) else [data]
    return ingest_events_list(events)
