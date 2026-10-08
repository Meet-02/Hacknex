"""
tools/test_retrieval.py
CLI test utility for Member 2 CVEvent Metadata Retrieval.
Searches through Member 1's structured CVEvent metadata only.

Usage:
    python tools/test_retrieval.py "<query>"
    python tools/test_retrieval.py "<query>" --json
    python tools/test_retrieval.py "<query>" --top_k 10
"""

import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events.event_search import search_events


def print_event_result(res: dict):
    print("=" * 70)
    print(f"QUERY:        \"{res.get('query')}\"")
    print(f"STATUS:       {str(res.get('status')).upper()}")
    print(f"MATCHED:      {res.get('results_count', 0)} of {res.get('total_events_in_db', 0)} indexed CVEvents")
    print("-" * 70)

    events = res.get("events", [])
    if not events:
        print("No matching events found in vector database.")
        if res.get("message"):
            print("Note: " + res["message"])
        print("=" * 70)
        return

    for evt in events:
        rank = evt.get("rank")
        evt_id = evt.get("event_id")
        sim = evt.get("similarity", 0.0)
        desc = evt.get("description")
        cam = evt.get("camera_id")
        loc = evt.get("location")
        t_start = evt.get("time_start")
        t_end = evt.get("time_end")
        rel = evt.get("relationship")
        ev = evt.get("evidence", {})

        print(f"\n  [Rank {rank}] Event ID: {evt_id} | Similarity: {sim:.3f}")
        print(f"    Description:  {desc}")
        print(f"    Location:     Camera {cam} ({loc}) | Time: {t_start} -> {t_end}")
        if rel:
            print(f"    Relationship: {rel}")
        print(f"    Frame Path:   {ev.get('frame_path')}")
        print(f"    Clip Path:    {ev.get('clip_path')}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test Member 1 CVEvent Retrieval")
    parser.add_argument("query", nargs="?", default=None, help="Natural language query to test")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return (default: 5)")
    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        sys.exit(0)

    res = search_events(args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print_event_result(res)


if __name__ == "__main__":
    main()
