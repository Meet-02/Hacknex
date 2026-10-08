"""
events/event_store.py
Structured event metadata storage and retrieval for Member 1 CV Events.
Stores event records in a persistent structured database (SQLite) without
generating or requiring vector embeddings.

Ready for connection with future video embedding pipeline via event_id.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _get_db_path() -> str:
    path = getattr(config, "EVENTS_DB_PATH", None)
    if not path:
        path = os.path.join(getattr(config, "PROJECT_DIR", "."), "data", "events_store.db")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return str(path)


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize structured events storage table."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                camera_id TEXT,
                location TEXT,
                timestamp_sec REAL,
                time_start TEXT,
                time_end TEXT,
                person_track_id INTEGER,
                object_class TEXT,
                object_confidence REAL,
                association_state TEXT,
                relationship TEXT,
                relationship_conf REAL,
                has_person INTEGER,
                has_bag INTEGER,
                roi_alert INTEGER,
                suspicious_object_alert INTEGER,
                frame_path TEXT,
                crop_path TEXT,
                clip_path TEXT,
                description TEXT,
                search_text TEXT,
                semantic_tags TEXT,
                raw_json TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp_sec)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_obj_class ON events(object_class)")
        conn.commit()


# Initialize on import
init_db()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    raw_json_str = row["raw_json"]
    raw_dict = json.loads(raw_json_str) if raw_json_str else {}
    semantic_tags = []
    if row["semantic_tags"]:
        try:
            semantic_tags = json.loads(row["semantic_tags"])
        except Exception:
            semantic_tags = []

    return {
        "event_id": row["event_id"],
        "camera_id": row["camera_id"],
        "location": row["location"],
        "timestamp_sec": row["timestamp_sec"],
        "time_start": row["time_start"],
        "time_end": row["time_end"],
        "person_track_id": row["person_track_id"],
        "object_class": row["object_class"],
        "object_confidence": row["object_confidence"],
        "association_state": row["association_state"],
        "relationship": row["relationship"],
        "relationship_conf": row["relationship_conf"],
        "has_person": bool(row["has_person"]),
        "has_bag": bool(row["has_bag"]),
        "roi_alert": bool(row["roi_alert"]),
        "suspicious_object_alert": bool(row["suspicious_object_alert"]),
        "frame_path": row["frame_path"],
        "crop_path": row["crop_path"],
        "clip_path": row["clip_path"],
        "description": row["description"],
        "search_text": row["search_text"],
        "semantic_tags": semantic_tags,
        "raw_event": raw_dict,
    }


def upsert_events(records: List[Dict[str, Any]]) -> None:
    """
    Upsert structured event records into storage.
    ZERO embeddings are generated or stored.
    
    records: list of dicts with:
        "id"        : str (event_id)
        "metadata"  : dict (scalar values and tags)
        "document"  : str (canonical description)
        "raw_event" : dict (optional original event payload)
    """
    if not records:
        return

    init_db()
    with _get_connection() as conn:
        for r in records:
            evt_id = r["id"]
            meta = r.get("metadata", {})
            doc = r.get("document", "")
            raw = r.get("raw_event") or {}

            # Fall back to raw event values if not present in metadata
            cam_info = raw.get("camera", {}) if isinstance(raw, dict) else {}
            time_info = raw.get("time", {}) if isinstance(raw, dict) else {}
            entities = raw.get("entities", {}) if isinstance(raw, dict) else {}
            person = entities.get("person", {}) if isinstance(entities, dict) else {}
            obj = entities.get("object", {}) if isinstance(entities, dict) else {}
            rel = raw.get("relationship", {}) if isinstance(raw, dict) else {}
            ev = raw.get("evidence", {}) if isinstance(raw, dict) else {}

            camera_id = str(meta.get("camera_id") or cam_info.get("camera_id") or raw.get("camera_id") or "")
            location = str(meta.get("location") or cam_info.get("location") or "")
            time_start = str(meta.get("time_start") or time_info.get("start") or "")
            time_end = str(meta.get("time_end") or time_info.get("end") or "")
            timestamp_sec = float(meta.get("timestamp_sec") or time_info.get("timestamp_sec") or raw.get("timestamp", 0.0) or 0.0)

            person_track_id = int(meta.get("person_track_id") if meta.get("person_track_id") is not None
                                  else (person.get("track_id") if person.get("track_id") is not None else -1))
            object_class = str(meta.get("object_class") or obj.get("class") or obj.get("class_name") or "")
            object_conf = float(meta.get("object_confidence") or obj.get("confidence") or obj.get("detection_confidence") or 0.0)
            assoc_state = str(meta.get("association_state") or obj.get("association_state") or "")

            relationship = str(meta.get("relationship") or rel.get("type") or "")
            rel_conf = float(meta.get("relationship_conf") or rel.get("confidence") or 0.0)

            has_person = 1 if meta.get("has_person", bool(person or raw.get("person"))) else 0
            has_bag = 1 if meta.get("has_bag", object_class in {"handbag", "backpack", "suitcase", "bag", "purse"}) else 0
            roi_alert = 1 if meta.get("roi_alert", raw.get("roi_alert", False)) else 0
            suspicious_alert = 1 if meta.get("suspicious_object_alert", raw.get("alert", False)) else 0

            frame_path = str(meta.get("frame_path") or ev.get("frame_path") or raw.get("evidence_frame_path") or "")
            crop_path = str(meta.get("crop_path") or ev.get("crop_path") or raw.get("evidence_crop_path") or "")
            clip_path = str(meta.get("clip_path") or ev.get("clip_path") or raw.get("clip_path") or "")

            description = str(meta.get("description") or doc or raw.get("description") or "")
            search_text = str(raw.get("search_text") or "")
            semantic_tags = raw.get("semantic_tags") or []
            tags_json = json.dumps(semantic_tags if isinstance(semantic_tags, list) else [str(semantic_tags)])
            raw_json = json.dumps(raw)

            conn.execute("""
                INSERT INTO events (
                    event_id, camera_id, location, timestamp_sec,
                    time_start, time_end, person_track_id, object_class,
                    object_confidence, association_state, relationship,
                    relationship_conf, has_person, has_bag, roi_alert,
                    suspicious_object_alert, frame_path, crop_path, clip_path,
                    description, search_text, semantic_tags, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    camera_id=excluded.camera_id,
                    location=excluded.location,
                    timestamp_sec=excluded.timestamp_sec,
                    time_start=excluded.time_start,
                    time_end=excluded.time_end,
                    person_track_id=excluded.person_track_id,
                    object_class=excluded.object_class,
                    object_confidence=excluded.object_confidence,
                    association_state=excluded.association_state,
                    relationship=excluded.relationship,
                    relationship_conf=excluded.relationship_conf,
                    has_person=excluded.has_person,
                    has_bag=excluded.has_bag,
                    roi_alert=excluded.roi_alert,
                    suspicious_object_alert=excluded.suspicious_object_alert,
                    frame_path=excluded.frame_path,
                    crop_path=excluded.crop_path,
                    clip_path=excluded.clip_path,
                    description=excluded.description,
                    search_text=excluded.search_text,
                    semantic_tags=excluded.semantic_tags,
                    raw_json=excluded.raw_json
            """, (
                evt_id, camera_id, location, timestamp_sec,
                time_start, time_end, person_track_id, object_class,
                object_conf, assoc_state, relationship, rel_conf,
                has_person, has_bag, roi_alert, suspicious_alert,
                frame_path, crop_path, clip_path, description,
                search_text, tags_json, raw_json
            ))
        conn.commit()


def upsert_event(event: Dict[str, Any], canonical_text: str = "") -> None:
    """
    Upsert a single CVEvent into structured storage.
    ZERO embeddings are generated.
    """
    from events.canonical_serializer import serialize_event_to_text

    description = canonical_text or serialize_event_to_text(event)
    rec = {
        "id": event["event_id"],
        "metadata": {
            "description": description,
        },
        "document": description,
        "raw_event": event,
    }
    upsert_events([rec])


def count_events() -> int:
    """Return count of indexed events in structured storage."""
    init_db()
    with _get_connection() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM events")
        return int(cursor.fetchone()[0])


def get_event(event_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single event by event_id."""
    init_db()
    with _get_connection() as conn:
        cursor = conn.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
        row = cursor.fetchone()
        return _row_to_dict(row) if row else None


def get_all_events() -> List[Dict[str, Any]]:
    """Retrieve all indexed events."""
    init_db()
    with _get_connection() as conn:
        cursor = conn.execute("SELECT * FROM events")
        return [_row_to_dict(row) for row in cursor.fetchall()]


def query_events(
    query_vector: Any = None,
    top_k: int = 5,
    where: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Query events using structured metadata filters.
    (Kept for backward compatibility with existing interfaces; requires ZERO embeddings).
    """
    events = get_all_events()
    if where:
        # Simple metadata filtering matching Chroma where dicts
        filtered = []
        for e in events:
            matches = True
            for k, v in where.items():
                if k == "$and" and isinstance(v, list):
                    for sub in v:
                        for sk, sv in sub.items():
                            val = sv.get("$eq", sv) if isinstance(sv, dict) else sv
                            if e.get(sk) != val:
                                matches = False
                else:
                    val = v.get("$eq", v) if isinstance(v, dict) else v
                    if e.get(k) != val:
                        matches = False
            if matches:
                filtered.append(e)
        events = filtered

    selected = events[:top_k]
    return {
        "ids": [e["event_id"] for e in selected],
        "distances": [0.0 for _ in selected],
        "metadatas": [
            {
                "camera_id": e["camera_id"],
                "location": e["location"],
                "time_start": e["time_start"],
                "time_end": e["time_end"],
                "timestamp_sec": e["timestamp_sec"],
                "relationship": e["relationship"],
                "frame_path": e["frame_path"],
                "crop_path": e["crop_path"],
                "clip_path": e["clip_path"],
            }
            for e in selected
        ],
        "documents": [e["description"] for e in selected],
    }


class EventsCollectionFacade:
    """Facade for backward compatibility with code calling get_events_collection()."""
    def count(self) -> int:
        return count_events()

    def get(self, limit: int = 10, include: Optional[List[str]] = None) -> Dict[str, Any]:
        events = get_all_events()[:limit]
        return {
            "ids": [e["event_id"] for e in events],
            "documents": [e["description"] for e in events],
            "metadatas": [
                {
                    "camera_id": e["camera_id"],
                    "location": e["location"],
                    "time_start": e["time_start"],
                    "time_end": e["time_end"],
                    "timestamp_sec": e["timestamp_sec"],
                }
                for e in events
            ]
        }


_FACADE = EventsCollectionFacade()


def get_events_collection() -> EventsCollectionFacade:
    """Return facade collection object supporting .count() and .get()."""
    return _FACADE


def get_client():
    """Stub client helper."""
    return None
