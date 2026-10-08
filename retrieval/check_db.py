"""Quick script to check what's in ChromaDB"""
from events.event_store import get_events_collection

col = get_events_collection()
res = col.get(limit=10, include=['documents', 'metadatas'])

print('=' * 70)
print('ChromaDB Contents')
print('=' * 70)
print(f'Total events indexed: {col.count()}\n')

print('First 3 events:\n')
for i in range(min(3, len(res['ids']))):
    print(f'{i+1}. Event ID: {res["ids"][i]}')
    print(f'   Description: {res["documents"][i]}')
    print(f'   Camera: {res["metadatas"][i].get("camera_id")}')
    print(f'   Location: {res["metadatas"][i].get("location")}')
    print(f'   Time: {res["metadatas"][i].get("time_start")}')
    print()
