"""Index the shared CV event contract into structured storage without embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "retrieval"))

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from events.index_events import ingest_event  # noqa: E402
from events.event_store import count_events  # noqa: E402
import config  # noqa: E402

DEFAULT_EVENTS = ROOT / "data" / "cv_events" / "detections.json"


def load_events(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    events = data if isinstance(data, list) else [data]
    return events


def validate_event(event: dict) -> bool:
    """Validate that an event complies with the contract."""
    if not isinstance(event, dict):
        return False
    if not event.get("event_id"):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, default=None,
                        help="CV event JSON (defaults to data/cv_events/detections.json)")
    parser.add_argument("--file", "-f", type=Path, default=None,
                        help="Alias for --input")
    args = parser.parse_args()

    input_path = args.input or args.file or DEFAULT_EVENTS
    path = input_path if input_path.is_absolute() else ROOT / input_path
    if not path.is_file():
        parser.error(f"CV event file not found: {path}. Run CV with --save-json first.")

    events = load_events(path)
    if not events:
        parser.error(f"No events found in {path}")

    valid_events = 0
    failed = []
    indexed = 0

    for event in events:
        if not validate_event(event):
            failed.append((event.get("event_id", "unknown"), "Invalid event schema or missing event_id"))
            continue
        valid_events += 1
        try:
            ingest_event(event)
            indexed += 1
        except Exception as exc:
            failed.append((event.get("event_id"), str(exc)))
            print(f"[failed] {event.get('event_id')}: {exc}", file=sys.stderr)

    embeddings_generated = 0

    print("[CV→RAG]")
    print(f"Loaded events: {len(events)}")
    print(f"Valid events: {valid_events}")
    print(f"Embeddings generated: {embeddings_generated}")
    print(f"Events indexed: {indexed}")
    if failed:
        print(f"Failed events: {len(failed)}")
    print(f"Unique event IDs in store: {count_events()}")
    print(f"Storage: {getattr(config, 'EVENTS_DB_PATH', ROOT / 'data' / 'events_store.db')}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
