"""
enricher.py

Deterministic event-enrichment for the Hacknex CV pipeline.

Produces three semantic fields from existing CV metadata — no LLM required:

    build_description(event)  → human-readable natural-language description
    build_semantic_tags(event)→ list of search-friendly tag strings
    build_search_text(event)  → single string optimised for embedding / BM25

Also handles suspicious-object / alert logic:

    check_suspicious_support(model_names) → warns about unsupported alert classes
    build_alert_fields(candidate, camera_id) → alert sub-dict for the event JSON

Design principles
─────────────────
- No facts are invented.  Every sentence is derived from values already in the
  event dict or ObjectCandidate.
- Bounding-box coordinates are kept in the structured JSON only; they are never
  put into natural-language text.
- "potential" is always used for suspicious objects — never "confirmed threat".
- All functions are pure (no side effects) so the RAG/backend team can import
  this module independently of pipeline.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from config import (
    ALERT_MIN_EVIDENCE_SCORE,
    OBJECT_SYNONYMS,
    SUSPICIOUS_OBJECTS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Startup check: warn about configured suspicious classes the model can't see
# ─────────────────────────────────────────────────────────────────────────────
def check_suspicious_support(model_class_names: dict) -> set:
    """
    Compare SUSPICIOUS_OBJECTS against the model's supported class names.
    Prints a WARNING for each unsupported class.
    Returns the set of suspicious class names the model CAN detect.

    Call once at pipeline startup after loading the detector.

    model_class_names : dict of {int: str} as returned by YOLO().names
    """
    known = set(model_class_names.values())
    supported = set()
    for cls_name in SUSPICIOUS_OBJECTS:
        if cls_name in known:
            supported.add(cls_name)
        else:
            print(f"[enricher] WARNING: suspicious object '{cls_name}' is configured "
                  f"but is NOT supported by the current detection model. "
                  f"Alerts for this class will not be generated.")
    if supported:
        print(f"[enricher] Suspicious-object alerting active for: "
              f"{', '.join(sorted(supported))}")
    return supported


# ─────────────────────────────────────────────────────────────────────────────
# Alert field builder
# ─────────────────────────────────────────────────────────────────────────────
def build_alert_fields(class_name: str,
                       evidence_score: float,
                       association_state: str,
                       supported_suspicious: set) -> dict:
    """
    Decide whether this object constitutes an alert.

    Conditions for alert=True:
      1. class_name is in SUSPICIOUS_OBJECTS
      2. class_name is actually supported by the loaded model
      3. association_state is "confirmed" OR "probable"
         (candidates are never alerts)
      4. evidence_score >= ALERT_MIN_EVIDENCE_SCORE

    Returns a dict with keys:
        alert, alert_type, alert_object, alert_label, alert_severity
    """
    null_alert = {
        "alert":          False,
        "alert_type":     None,
        "alert_object":   None,
        "alert_label":    None,
        "alert_severity": None,
    }

    if class_name not in SUSPICIOUS_OBJECTS:
        return null_alert
    if class_name not in supported_suspicious:
        return null_alert
    if association_state not in ("confirmed", "probable"):
        return null_alert
    if evidence_score < ALERT_MIN_EVIDENCE_SCORE:
        return null_alert

    cfg = SUSPICIOUS_OBJECTS[class_name]
    return {
        "alert":          True,
        "alert_type":     "suspicious_object",
        "alert_object":   class_name,
        "alert_label":    cfg["alert_label"],
        "alert_severity": cfg["severity"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Natural-language description
# ─────────────────────────────────────────────────────────────────────────────
def _cam_num(camera_id: str) -> str:
    """'cam_03' → 'camera 3', 'cam_03b' → 'cam_03b'."""
    parts = camera_id.split("_")
    if len(parts) == 2 and parts[1].isdigit():
        return f"camera {int(parts[1])}"
    return camera_id


def build_description(event: dict) -> str:
    """
    Build a moderately detailed, human-readable description from the event dict.
    The event must already contain all structured fields (including alert fields).
    No coordinates are included; all numeric scores are rounded to 2 d.p.
    """
    cam      = _cam_num(event["camera_id"])
    ts       = event["timestamp"]
    person   = event.get("person", {})
    track_id = person.get("track_id")
    p_conf   = person.get("confidence")
    obj      = event.get("object", {})
    cls      = obj.get("class_name", "")
    det_conf = obj.get("detection_confidence")
    spatial  = obj.get("spatial_score")
    temporal = obj.get("temporal_score")
    evidence = obj.get("evidence_score", event.get("evidence_score"))
    state    = obj.get("association_state", event.get("association_state", ""))
    source   = obj.get("source", "")
    alert      = event.get("alert", False)
    severity   = event.get("alert_severity")
    inside_roi = event.get("inside_roi", False)
    roi_phrase = " inside the configured ROI" if inside_roi else ""

    if event.get("roi_alert"):
        actor = (f"A person with track ID {track_id}" if cls == "person" and track_id is not None
                 else "A person" if cls == "person" else f"A {cls}")
        return (f"{actor} entered the configured danger-zone ROI on {cam} at {ts:.2f} seconds. "
                f"The ROI intrusion is flagged as a {event.get('roi_alert_severity', 'high')}-severity "
                f"danger event. Evidence references the original CCTV frame and clip.")

    tid_str  = f"track ID {track_id}" if track_id is not None else "an untracked person"

    # ── Alert / suspicious-object description ─────────────────────────────
    if alert and cls in SUSPICIOUS_OBJECTS:
        cfg = SUSPICIOUS_OBJECTS[cls]
        parts = [
            f"A potential {cls} was detected on {cam} at {ts:.2f} seconds "
            f"near person {tid_str}{roi_phrase}.",
        ]
        if det_conf is not None:
            parts.append(f"The {cls} was detected with {det_conf:.2f} confidence.")
        if temporal is not None:
            parts.append(f"The temporal consistency score is {temporal:.2f}")
            if evidence is not None:
                parts[-1] += f" and the overall evidence score is {evidence:.2f}."
            else:
                parts[-1] += "."
        elif evidence is not None:
            parts.append(f"The overall evidence score is {evidence:.2f}.")
        parts.append(
            f"The suspicious-object detection is {state} as an alert event "
            f"with {severity} severity."
        )
        return " ".join(parts)

    # ── No object ─────────────────────────────────────────────────────────
    if not cls:
        person_conf_str = (f" The person detection confidence is {p_conf:.2f}."
                           if p_conf is not None else "")
        return (f"A person with {tid_str} is visible on {cam} at {ts:.2f} seconds{roi_phrase}."
                f"{person_conf_str} No associated object was confirmed for this event.")

    # ── Normal object description ─────────────────────────────────────────
    parts = [f"A person with {tid_str} is visible on {cam} at {ts:.2f} seconds{roi_phrase}."]

    # State-sensitive opening sentence
    if state == "confirmed":
        parts.append(
            f"The person is associated with a {cls} detected with "
            f"{det_conf:.2f} confidence." if det_conf is not None else
            f"The person is associated with a {cls}."
        )
    elif state == "probable":
        parts.append(
            f"A {cls} is detected near the person with "
            f"{det_conf:.2f} confidence." if det_conf is not None else
            f"A {cls} is detected near the person."
        )
    else:
        parts.append(
            f"A {cls} is a candidate object near the person"
            + (f" with {det_conf:.2f} confidence." if det_conf is not None else ".")
        )

    # Scores
    score_parts = []
    if spatial is not None:
        score_parts.append(f"The spatial association score is {spatial:.2f}")
    if temporal is not None:
        score_parts.append(f"the temporal consistency score is {temporal:.2f}")
    if score_parts:
        parts.append(" and ".join(score_parts) + ".")

    if evidence is not None:
        if state == "confirmed":
            parts.append(
                f"The overall evidence score is {evidence:.2f}, "
                f"and the association is confirmed."
            )
        elif state == "probable":
            parts.append(
                f"The overall evidence score is {evidence:.2f}, so the "
                f"object-person association is currently probable rather than confirmed."
            )
        else:
            parts.append(f"The overall evidence score is {evidence:.2f}.")

    # Source
    if source == "crop":
        parts.append("The detection was obtained from a second-pass person-crop image.")
    elif source == "fullframe":
        parts.append("The detection was obtained from the full-frame image.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic tags
# ─────────────────────────────────────────────────────────────────────────────
def build_semantic_tags(event: dict) -> list:
    """
    Build a flat list of search-friendly tag strings deterministically from the event.
    Only facts present in the event are used — no invented relationships.
    """
    tags = []
    cam_id   = event["camera_id"]
    person   = event.get("person", {})
    obj      = event.get("object", {})
    cls      = obj.get("class_name", "")
    state    = obj.get("association_state", "")
    alert    = event.get("alert", False)
    severity = event.get("alert_severity", event.get("roi_alert_severity"))

    # Camera
    parts = cam_id.split("_")
    if len(parts) == 2 and parts[1].isdigit():
        tags.append(f"camera_{int(parts[1])}")
    else:
        tags.append(cam_id)

    # Person/object identity
    if event.get("roi_alert") and cls != "person":
        tags.append("object")
    else:
        tags.append("person")
    if cls == "person" and person.get("track_id") is not None:
        tags.append(f"person_track_{person['track_id']}")

    # Object
    if cls:
        tags.append(cls)
        # Synonyms — only those that don't assert unverified facts
        for syn in OBJECT_SYNONYMS.get(cls, []):
            tags.append(syn)

    # Association state
    if state:
        tags.append(state)
        if state in ("confirmed", "probable"):
            tags.append("person_object_association")
        if state == "confirmed" and cls and cls not in SUSPICIOUS_OBJECTS:
            tags.append("carrying")  # only for non-suspicious confirmed associations

    # Alert
    if alert:
        tags.append("alert")
        tags.append("suspicious_object")
        if severity:
            tags.append(f"{severity}_severity")

    # ROI
    if event.get("inside_roi") is True:
        tags.append("inside_roi")
    if event.get("roi_alert"):
        tags.extend(["roi", "danger", "danger_zone", "restricted_zone", "roi_intrusion", "roi_entry", "alert"])
        if severity:
            tags.append(f"{severity}_severity")

    return tags


# ─────────────────────────────────────────────────────────────────────────────
# Search text
# ─────────────────────────────────────────────────────────────────────────────
def build_search_text(event: dict) -> str:
    """
    Build a single flat string for embedding / BM25 retrieval.
    Prioritises semantic concepts over raw numbers.
    """
    cam_id   = event["camera_id"]
    ts       = event["timestamp"]
    person   = event.get("person", {})
    obj      = event.get("object", {})
    cls      = obj.get("class_name", "")
    state    = obj.get("association_state", "")
    evidence = obj.get("evidence_score", event.get("evidence_score"))
    alert    = event.get("alert", False)
    severity = event.get("alert_severity", event.get("roi_alert_severity"))

    cam_num  = _cam_num(cam_id)
    track_id = person.get("track_id")
    tid_str  = f"person track {track_id}" if track_id is not None else "person"

    inside_roi = event.get("inside_roi", False)

    if event.get("roi_alert"):
        actor = (f"person track {track_id}" if cls == "person" and track_id is not None
                 else "person" if cls == "person" else cls)
        return " ".join(t for t in [
            cam_num, actor, "entered danger zone", "ROI intrusion", "inside ROI",
            f"{event.get('roi_alert_severity', 'high')} severity alert",
            f"timestamp {ts:.2f} seconds",
        ] if t)

    tokens = []

    if alert and cls in SUSPICIOUS_OBJECTS:
        tokens += [
            cam_num,
            tid_str,
            f"potential {cls}",
            "suspicious object",
        ]
        tokens += OBJECT_SYNONYMS.get(cls, [])
        if state:
            tokens.append(state)
        if inside_roi:
            tokens.append("inside ROI")
        if severity:
            tokens.append(f"{severity} severity alert")
        tokens.append(f"timestamp {ts:.2f} seconds")
        if evidence is not None:
            if evidence >= 0.85:
                tokens.append("high evidence")
            elif evidence >= 0.70:
                tokens.append("moderate evidence")
    else:
        tokens += [cam_num, tid_str]
        if cls:
            tokens.append(cls)
            tokens += OBJECT_SYNONYMS.get(cls, [])
        if state in ("confirmed", "probable"):
            tokens += [state, "person object association"]
            if state == "confirmed" and cls not in SUSPICIOUS_OBJECTS:
                tokens.append("carrying")
        if inside_roi:
            tokens.append("inside ROI")
        tokens.append(f"timestamp {ts:.2f} seconds")
        if evidence is not None:
            if evidence >= 0.85:
                tokens.append("high evidence person object association")
            elif evidence >= 0.70:
                tokens.append("moderate evidence person object association")

    return " ".join(t for t in tokens if t)


# ─────────────────────────────────────────────────────────────────────────────
# Real-world date/time attachment (Section 14)
# ─────────────────────────────────────────────────────────────────────────────
def attach_datetime_fields(event: dict,
                            recording_start: Optional[str] = None,
                            timezone: Optional[str] = None) -> dict:
    """
    Attach real-world datetime fields if recording start metadata is available.
    If recording start metadata is NOT available:
      event_time, event_date, event_clock_time, timezone are null and datetime_available=False.
    Never invent date/time.
    """
    if not recording_start:
        event["event_time"]        = None
        event["event_date"]        = None
        event["event_clock_time"]  = None
        event["timezone"]          = timezone
        event["datetime_available"]= False
        return event

    try:
        dt_start = datetime.fromisoformat(recording_start)
        ts = event.get("timestamp", 0.0)
        event_dt = dt_start + timedelta(seconds=ts)
        event["event_time"]        = event_dt.isoformat()
        event["event_date"]        = event_dt.strftime("%Y-%m-%d")
        event["event_clock_time"]  = event_dt.strftime("%H:%M:%S.%f")[:-3]
        event["timezone"]          = timezone or (event_dt.tzname() or None)
        event["datetime_available"]= True
    except Exception:
        event["event_time"]        = None
        event["event_date"]        = None
        event["event_clock_time"]  = None
        event["timezone"]          = timezone
        event["datetime_available"]= False

    return event


# ─────────────────────────────────────────────────────────────────────────────
# Main enrichment entry point
# ─────────────────────────────────────────────────────────────────────────────
def enrich_event(event: dict,
                 supported_suspicious: set,
                 recording_start: Optional[str] = None,
                 timezone: Optional[str] = None) -> dict:
    """
    Add alert fields, description, semantic_tags, search_text, association
    sub-dict, and datetime fields to an event dict in-place. Returns the same dict.

    Call this AFTER the event dict has been fully built (all structured fields
    present), but BEFORE writing to JSON.
    """
    obj   = event.get("object", {})
    cls   = obj.get("class_name", "")
    score = obj.get("evidence_score", 0.0)
    state = obj.get("association_state", "")

    # Convenience aliases for Member 2 / Member 3 backward compatibility
    event["frame"] = event.get("frame_number", event.get("frame", 0))
    if "person" in event and "track_id" in event["person"]:
        event["person_track_id"] = event["person"]["track_id"]

    if obj:
        if "class" not in obj:
            obj["class"] = obj.get("class_name")
        if "confidence" not in obj:
            obj["confidence"] = obj.get("detection_confidence")

    event["association"] = {
        "state":          state,
        "spatial_score":  obj.get("spatial_score", 0.0),
        "temporal_score": obj.get("temporal_score", 0.0),
        "evidence_score": score,
        "seg_score":      obj.get("seg_score", 0.0),
    }

    # Alert fields
    alert_fields = build_alert_fields(cls, score, state, supported_suspicious)
    event.update(alert_fields)

    # Real-world datetime metadata
    attach_datetime_fields(event, recording_start, timezone)

    # Semantic enrichment
    event["description"]   = build_description(event)
    event["semantic_tags"] = build_semantic_tags(event)
    event["search_text"]   = build_search_text(event)

    return event
