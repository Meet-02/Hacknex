"""
run_chat.py
Complete pipeline: Ingest detections.json → Start RAG chat interface
"""

import sys
import os

try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def main():
    print("=" * 70)
    print("VISION-IQ Member 2: RAG Chat System")
    print("=" * 70)
    print()
    
    # Step 1: Check if detections are already indexed
    from events.event_store import count_events
    
    event_count = count_events()
    print(f"[Status] ChromaDB currently has {event_count} indexed events")
    
    if event_count == 0:
        print("\n[Setup] No events found. Indexing detections.json...")
        print()
        
        # Step 2: Ingest detections.json
        from events.ingest_detections import ingest_detections_file
        
        detections_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "metadata",
            "detections.json"
        )
        
        if not os.path.exists(detections_path):
            print(f"[Error] detections.json not found at: {detections_path}")
            print("Please place your detections.json file in the metadata/ folder")
            sys.exit(1)
        
        try:
            processed, skipped = ingest_detections_file(detections_path)
            print(f"\n[Success] Indexed {processed} events from detections.json")
            if skipped > 0:
                print(f"[Warning] {skipped} events were skipped due to errors")
        except Exception as e:
            print(f"\n[Error] Failed to ingest detections: {e}")
            sys.exit(1)
    else:
        print(f"[Status] Using existing {event_count} indexed events")
    
    print()
    print("=" * 70)
    print("Starting interactive chat...")
    print("=" * 70)
    print()
    
    # Step 3: Start chat interface
    from rag.chat import interactive_chat
    interactive_chat()


if __name__ == "__main__":
    main()
