"""
events/ingest_detections.py
Ingests Member 1's detections.json file and indexes events into structured storage.
ZERO embeddings are generated.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import List, Dict, Any
from tqdm import tqdm
from events.canonical_serializer import serialize_event
from events.event_store import upsert_event
import config


def format_timestamp(seconds: float) -> str:
    """Convert timestamp in seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def convert_detection_to_event(detection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert detection.json format to CVEvent format.
    
    Input format:
    {
      "event_id": "uuid",
      "camera_id": "cam_03",
      "frame_number": 207,
      "timestamp": 3.455,
      "person": {"track_id": 2, "confidence": 0.911, "bbox": [...]},
      "object": {
        "class_name": "handbag",
        "detection_confidence": 0.691,
        "association_state": "confirmed",
        ...
      },
      "evidence_frame_path": "output/frames/...",
      "evidence_crop_path": "output/crops/..."
    }
    
    Output: CVEvent format for serialization
    """
    camera_id = detection.get("camera_id", "unknown")
    location = config.get_camera_location(camera_id)
    timestamp_sec = detection.get("timestamp", 0.0)
    timestamp_str = format_timestamp(timestamp_sec)
    
    person_data = detection.get("person", {})
    object_data = detection.get("object", {})
    
    # Build CVEvent structure
    obj = dict(object_data)
    if "class" not in obj and obj.get("class_name"):
        obj["class"] = obj["class_name"]
    if "confidence" not in obj and obj.get("detection_confidence") is not None:
        obj["confidence"] = obj["detection_confidence"]
    event = {
        "event_id": detection["event_id"],
        "event_type": detection.get("event_type", ""),
        "camera": {
            "camera_id": camera_id,
            "location": location
        },
        "time": {
            "start": timestamp_str,
            "end": timestamp_str,  # Single frame event
            "timestamp_sec": timestamp_sec,
        },
        "entities": {
            "person": {
                "track_id": person_data.get("track_id"),
                "class": "person",
                "confidence": person_data.get("confidence")
            },
            "object": obj
        },
        "association": detection.get("association", {}),
        "relationship": detection.get("relationship", {}),
        "description": detection.get("description", ""),
        "search_text": detection.get("search_text", ""),
        "semantic_tags": detection.get("semantic_tags", []),
        "roi": detection.get("roi", {}),
        "roi_alert": detection.get("roi_alert", False),
        "suspicious_object_alert": detection.get("alert", False),
        "evidence": {
            "frame_path": detection.get("frame_path") or detection.get("evidence_frame_path"),
            "crop_path": detection.get("crop_path") or detection.get("evidence_crop_path"),
            "clip_path": detection.get("clip_path"),
            "frame_number": detection.get("frame_number"),
        }
    }
    
    return event


def ingest_detections_file(file_path: str, batch_size: int = 50):
    """
    Ingest detections.json file and index all events into ChromaDB.
    
    Args:
        file_path: Path to detections.json
        batch_size: Number of events to process before upserting (default: 50)
    """
    print(f"[Ingest] Loading detections from: {file_path}")
    
    with open(file_path, 'r') as f:
        detections = json.load(f)
    
    total = len(detections)
    print(f"[Ingest] Found {total} detection events")
    
    # Process in batches
    processed = 0
    skipped = 0
    
    for detection in tqdm(detections, desc="Indexing events"):
        try:
            # Convert detection format to CVEvent format
            event = convert_detection_to_event(detection)
            
            # Serialize to canonical text
            canonical_text = serialize_event(event)
            
            # Upsert into ChromaDB
            upsert_event(event, canonical_text)
            processed += 1
            
        except Exception as e:
            print(f"\n[Warning] Failed to process event {detection.get('event_id')}: {e}")
            skipped += 1
            continue
    
    print(f"\n[Ingest] ✓ Indexed {processed} events")
    if skipped > 0:
        print(f"[Ingest] ⚠ Skipped {skipped} events due to errors")
    
    return processed, skipped


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest Member 1's detections.json")
    parser.add_argument(
        "--file",
        type=str,
        default="metadata/detections.json",
        help="Path to detections.json (default: metadata/detections.json)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for processing (default: 50)"
    )
    
    args = parser.parse_args()
    
    file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.file
    )
    
    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        sys.exit(1)
    
    ingest_detections_file(file_path, args.batch_size)
