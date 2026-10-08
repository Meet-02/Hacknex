"""
events/canonical_serializer.py
Converts Member 1's structured CVEvent JSON into a natural canonical text description.
This description is embedded into vector space for semantic retrieval.
"""

from typing import Dict, Any


def serialize_event_to_text(event: Dict[str, Any]) -> str:
    """
    Generate a canonical natural-language description from a Member 1 CVEvent record.
    
    Example input:
    {
        "event_id": "evt_00123",
        "camera": {"camera_id": "cam_01", "location": "entrance"},
        "time": {"start": "15:30:21", "end": "15:30:25"},
        "event_type": "object_interaction",
        "entities": {
            "person": {"track_id": 17, "class": "person"},
            "object": {"track_id": 42, "class": "backpack", "attributes": {"color": "red"}}
        },
        "relationship": {"type": "carrying", "confidence": 0.86}
    }
    
    Returns:
        "Person 17 is carrying a red backpack (ID 42) in camera cam_01 at entrance. The event occurred between 15:30:21 and 15:30:25."
    """
    # Native Hacknex CV events already carry grounded natural-language fields.
    # Use those directly instead of fabricating a relationship or CVEvent data.
    if event.get("description") or event.get("search_text"):
        parts = [event.get("description", ""), event.get("search_text", "")]
        tags = event.get("semantic_tags", [])
        if tags:
            parts.append("Semantic tags: " + ", ".join(str(tag) for tag in tags))
        return " ".join(part.strip() for part in parts if part and part.strip())

    cam_info = event.get("camera", {})
    cam_id = cam_info.get("camera_id", "unknown camera")
    location = cam_info.get("location", "").replace("_", " ")

    time_info = event.get("time", {})
    start = time_info.get("start", "")
    end = time_info.get("end", "")

    entities = event.get("entities", {})
    person = entities.get("person", {})
    obj = entities.get("object", {})
    vehicle = entities.get("vehicle", {})
    rel = event.get("relationship", {})

    rel_type = rel.get("type", "").lower()
    event_type = event.get("event_type", "").lower()

    # Case 1: Person interacting with object (e.g. carrying backpack)
    if person and obj and rel_type:
        p_str = f"Person {person.get('track_id', '')}".strip()
        obj_class = obj.get("class", "object")
        attrs = obj.get("attributes", {})
        color = attrs.get("color", "")
        obj_desc = f"{color} {obj_class}".strip()
        obj_id = obj.get("track_id")
        obj_id_str = f" (ID {obj_id})" if obj_id is not None else ""

        desc = f"{p_str} is {rel_type} a {obj_desc}{obj_id_str}"
    
    # Case 2: Person motion (entered, exited, loitering)
    elif person and ("entered" in event_type or "exited" in event_type or "loitering" in event_type):
        p_str = f"Person {person.get('track_id', '')}".strip()
        action = event_type.replace("person_", "").replace("_", " ")
        desc = f"{p_str} {action}"

    # Case 3: Vehicle movement
    elif vehicle:
        v_class = vehicle.get("class", "vehicle")
        v_attrs = vehicle.get("attributes", {})
        v_color = v_attrs.get("color", "")
        v_desc = f"{v_color} {v_class}".strip()
        desc = f"{v_desc.capitalize()} entered"

    # Case 4: General entity detected
    elif person:
        desc = f"Person {person.get('track_id', '')} detected".strip()
    elif obj:
        desc = f"{obj.get('class', 'Object')} detected"
    else:
        desc = f"Event {event.get('event_id', '')} ({event_type.replace('_', ' ')})"

    # Append location
    loc_part = ""
    if location:
        loc_part = f" at {location}"
    if cam_id:
        loc_part += f" in camera {cam_id}"
    desc += loc_part

    # Append timing
    if start and end:
        desc += f". Occurred between {start} and {end}."
    elif start:
        desc += f" at {start}."
    else:
        desc += "."

    return desc.strip()


def serialize_event(event: Dict[str, Any]) -> str:
    """
    Alias for serialize_event_to_text for backward compatibility.
    """
    return serialize_event_to_text(event)
