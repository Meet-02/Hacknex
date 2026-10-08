"""Exercise CV event loading, existing RAG indexing, and natural-language search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "retrieval"))
sys.path.insert(0, str(ROOT / "scripts"))

from index_cv_events import DEFAULT_EVENTS, load_events  # noqa: E402
from events.index_events import ingest_event  # noqa: E402
from events.event_store import count_events  # noqa: E402
from events.event_search import search_events  # noqa: E402


def resolve_evidence(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="person carrying a handbag")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--file", type=Path, default=DEFAULT_EVENTS)
    args = parser.parse_args()
    path = args.file if args.file.is_absolute() else ROOT / args.file
    if not path.is_file():
        parser.error(f"CV event file not found: {path}")

    events = load_events(path)
    if not events:
        parser.error(f"No events found in {path}")
    print(f"Loaded {len(events)} CV events")

    failures = []
    for event in events:
        try:
            ingest_event(event)  # upsert is idempotent; confirms this contract indexes
        except Exception as exc:
            failures.append((event.get("event_id"), str(exc)))
    print(f"Index check: {len(events) - len(failures)} indexed, {len(failures)} failed")
    for event_id, error in failures:
        print(f"  Failed {event_id}: {error}")
    if failures:
        return 1

    print(f"Event store count: {count_events()}")
    print(f"Query: {args.query}")
    result = search_events(args.query, top_k=args.top_k)
    if not result.get("events"):
        print("No matching events returned.")
        return 1

    missing_paths = 0
    for hit in result["events"]:
        event = next((item for item in events if item["event_id"] == hit["event_id"]), {})
        print(f"event_id: {hit.get('event_id')}")
        print(f"camera_id: {hit.get('camera_id')}")
        print(f"timestamp: {event.get('timestamp', hit.get('time_start'))}")
        evidence = hit.get("evidence", {})
        for key in ("frame_path", "crop_path", "clip_path"):
            evidence_path = evidence.get(key) or event.get({
                "frame_path": "evidence_frame_path",
                "crop_path": "evidence_crop_path",
                "clip_path": "clip_path",
            }[key])
            if evidence_path:
                exists = resolve_evidence(evidence_path).is_file()
                print(f"{key}: {evidence_path} ({'exists' if exists else 'missing'})")
                if not exists:
                    missing_paths += 1
            else:
                print(f"{key}: unavailable")
        print(f"description: {hit.get('description')}")
        print()
    if missing_paths:
        print(f"Warning: {missing_paths} evidence path(s) could not be verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
