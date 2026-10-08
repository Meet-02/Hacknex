"""
rag/chat.py
RAG-based conversational interface for CCTV video intelligence.
Uses retrieved events to answer natural language questions.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from typing import List, Dict, Any
from events.event_search import search_events


def format_event_for_context(event: Dict[str, Any]) -> str:
    """Format a single event into human-readable context for RAG."""
    return (
        f"Event {event['event_id']}:\n"
        f"  - {event['description']}\n"
        f"  - Location: {event['location']} (Camera {event['camera_id']})\n"
        f"  - Time: {event['time_start']} to {event['time_end']}\n"
        f"  - Confidence: {event['similarity']:.2%}\n"
        f"  - Evidence: {event['evidence'].get('frame_path') or event['evidence'].get('crop_path')}\n"
    )


def build_rag_context(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search for relevant events and build RAG context.
    
    Args:
        query: User's natural language question
        top_k: Number of top events to retrieve
        
    Returns:
        Dict with:
            - query: Original query
            - context: Formatted text context for LLM
            - events: List of retrieved event metadata
            - summary: Quick summary stats
    """
    # Search for relevant events
    search_result = search_events(query, top_k=top_k)
    
    events = search_result.get("events", [])
    
    if not events:
        return {
            "query": query,
            "context": "No matching events were found in the video database.",
            "events": [],
            "summary": {
                "matched": 0,
                "total_in_db": search_result.get("total_events_in_db", 0)
            }
        }
    
    # Build context string for LLM
    context_parts = [
        f"Based on the CCTV footage database, here are {len(events)} relevant events:\n"
    ]
    
    for event in events:
        context_parts.append(format_event_for_context(event))
    
    context = "\n".join(context_parts)
    
    return {
        "query": query,
        "context": context,
        "events": events,
        "summary": {
            "matched": len(events),
            "total_in_db": search_result.get("total_events_in_db", 0),
            "cameras": list(set(e["camera_id"] for e in events)),
            "time_range": {
                "earliest": min(e["time_start"] for e in events),
                "latest": max(e["time_end"] for e in events)
            }
        }
    }


def chat(query: str, top_k: int = 5, format: str = "context") -> Any:
    """
    Main chat interface for RAG queries.
    
    Args:
        query: User's natural language question
        top_k: Number of events to retrieve
        format: Output format - "context" (RAG context) or "answer" (simple answer)
        
    Returns:
        - If format="context": Full RAG context dict
        - If format="answer": Human-readable answer string
    """
    rag_data = build_rag_context(query, top_k=top_k)
    
    if format == "context":
        return rag_data
    
    # format == "answer": Generate simple human-readable response
    events = rag_data["events"]
    summary = rag_data["summary"]
    
    if not events:
        return (
            f"I couldn't find any events matching '{query}' in the video database. "
            f"The database contains {summary['total_in_db']} total events."
        )
    
    # Group events by camera and time
    camera_groups = {}
    for event in events:
        cam = event["camera_id"]
        if cam not in camera_groups:
            camera_groups[cam] = []
        camera_groups[cam].append(event)
    
    # Build answer
    answer_parts = [
        f"I found {summary['matched']} matching events in the CCTV footage:\n"
    ]
    
    for camera, cam_events in camera_groups.items():
        location = cam_events[0]["location"]
        answer_parts.append(f"\n📹 Camera {camera} ({location}):")
        
        for event in cam_events:
            desc = event["description"]
            time = event["time_start"]
            confidence = event["similarity"]
            answer_parts.append(
                f"  • At {time}: {desc} (confidence: {confidence:.0%})"
            )
    
    # Add evidence paths
    answer_parts.append("\n📁 Evidence files:")
    for i, event in enumerate(events[:3], 1):  # Show top 3
        evidence = event["evidence"]
        frame = evidence.get("frame_path") or evidence.get("crop_path")
        if frame:
            answer_parts.append(f"  {i}. {frame}")
    
    return "\n".join(answer_parts)


def interactive_chat():
    """Start an interactive chat session."""
    print("=" * 70)
    print("VISION-IQ RAG Chat Interface")
    print("Ask questions about the CCTV footage database.")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 70)
    print()
    
    while True:
        try:
            query = input("You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            print()
            answer = chat(query, format="answer")
            print(f"Assistant: {answer}")
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VISION-IQ RAG Chat Interface")
    parser.add_argument("query", nargs="?", help="Single query (optional)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of events to retrieve")
    parser.add_argument("--format", choices=["context", "answer"], default="answer", help="Output format")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive chat session")
    
    args = parser.parse_args()
    
    if args.interactive or not args.query:
        interactive_chat()
    else:
        result = chat(args.query, top_k=args.top_k, format=args.format)
        
        if args.format == "context":
            import json
            print(json.dumps(result, indent=2))
        else:
            print(result)
