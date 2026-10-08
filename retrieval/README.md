# VISION-IQ — Member 2: CVEvent Metadata Retrieval & RAG

**HackNex 2026 — PS05: Multi-Stream Video Intelligence with Conversational Query**

This module answers:

> Given a natural-language query, find the most relevant **CVEvent metadata** from Member 1's structured event store and return it with **event_id, camera, timestamp, frame path, clip path, confidence, and reason**.

---

## Architecture Overview

**Member 2 is a text-based semantic search engine for structured CVEvent metadata.**

### Data Flow:
```
Member 1 (YOLO + Tracking)
    ↓
CVEvent JSON (entities, relationships, camera, timestamp)
    ↓
Member 2: Canonical Text Serializer
    "Person 17 is carrying a red handbag at main gate in camera G336"
    ↓
SigLIP-2 Text Encoder → 768-dim vector
    ↓
ChromaDB Vector Store (cctv_events collection)
    ↓
User Query: "people carrying bags"
    ↓
Vector Similarity Search + Metadata Filtering
    ↓
Matched Events with Evidence Paths
```

---

## Folder Structure

```
retrieval/
├── config.py                           # Model IDs, paths, thresholds, weights, camera map
├── models/siglip_model.py              # SigLIP2 text encoder (encode_text)
├── embeddings/text_embedding.py        # embed_query(q) → (768,) numpy
├── events/
│   ├── canonical_serializer.py         # CVEvent JSON → natural language description
│   ├── event_store.py                  # ChromaDB event collection management
│   ├── index_events.py                 # Ingests Member 1's CVEvent JSON into vector DB
│   └── event_search.py                 # search_events(query) → matched event metadata
├── vector_db/chroma_store.py           # ChromaDB CRUD + cosine similarity query
├── retrieval/
│   ├── query_parser.py                 # NL query → {object, attribute, location, time}
│   ├── metadata_filter.py              # parsed → Chroma where-clause
│   ├── semantic_search.py              # query → Top-K vector hits
│   ├── hybrid_search.py                # filter + semantic + auto-relax fallback
│   └── reranker.py                     # Multi-signal weighted reranker
├── rag/context_builder.py              # Evidence envelope + reason builder
├── evaluation/
│   ├── dataset.py                      # Labeled query set (positive + negative)
│   └── metrics.py                      # Recall@K, MRR, camera acc, latency
├── tools/test_retrieval.py             # CLI testing utility
├── chroma_db/                          # Persistent vector store (git-ignored)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
cd d:\VISION-IQ\retrieval
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Index Member 1's CVEvent data

Member 1 provides a JSON/JSONL file with CVEvent records:

```python
from events.index_events import ingest_events_from_file

# From JSON file
ingest_events_from_file("path/to/member1_events.json")

# Or from JSONL
ingest_events_from_file("path/to/member1_events.jsonl")
```

**CVEvent Format Expected:**
```json
{
  "event_id": "evt_00101",
  "camera": {"camera_id": "G336", "location": "main_gate"},
  "time": {"start": "00:00:01", "end": "00:00:05"},
  "entities": {
    "person": {"track_id": 17, "class": "person"},
    "object": {"track_id": 42, "class": "handbag", "attributes": {"color": "red"}}
  },
  "relationship": {"type": "carrying"},
  "evidence": {
    "frame_path": "data/frames/G336/00_00_02.jpg",
    "clip_path": "data/clips/G336/00_00_02.mp4"
  }
}
```

### 3. Run a query

**Option A: Python API**
```python
from events.event_search import search_events
import json

result = search_events("people carrying bags", top_k=5)
print(json.dumps(result, indent=2))
```

**Option B: Command Line**
```bash
.venv\Scripts\python.exe tools\test_retrieval.py "people carrying bags"
```

**Option C: JSON Output (for Member 3 Backend)**
```bash
.venv\Scripts\python.exe tools\test_retrieval.py "red handbag at main gate" --json
```

### 4. Run evaluation metrics

```bash
.venv\Scripts\python.exe evaluation\metrics.py
```

---

## Output Contract

### Success (Verified Match)

```json
{
  "status": "success",
  "query": "people carrying bags",
  "results_count": 3,
  "total_events_in_db": 3,
  "events": [
    {
      "rank": 1,
      "event_id": "evt_00101",
      "similarity": 0.697,
      "description": "Person 17 is carrying a red handbag (ID 42) at main gate in camera G336.",
      "camera_id": "G336",
      "location": "main_gate",
      "time_start": "00:00:01",
      "time_end": "00:00:05",
      "relationship": "carrying",
      "evidence": {
        "frame_path": "data/frames/G336/00_00_02.jpg",
        "clip_path": "data/clips/G336/00_00_02.mp4"
      }
    }
  ]
}
```

### No Match Found

```json
{
  "status": "success",
  "query": "yellow helicopter",
  "results_count": 0,
  "total_events_in_db": 3,
  "events": [],
  "message": "No events matched the query in the vector database."
}
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Text-only embedding** | Member 1 provides structured metadata. Embedding canonical text descriptions is faster, more precise, and enables explainability. |
| **ChromaDB** | Local embedded vector DB with zero infrastructure setup. Persistent storage in `chroma_db/` folder. |
| **SigLIP-2 Text Encoder** | State-of-the-art vision-language model for natural language semantic matching. |
| **Canonical Serialization** | Converts structured CVEvent JSON into human-readable sentences that preserve entity relationships. |
| **Cosine Similarity** | Standard metric for text embedding comparison (range: 0-1, higher = better match). |
| **No Visual Re-embedding** | Relies on Member 1's YOLO + tracking for object/attribute detection. Member 2 focuses purely on retrieval and ranking. |

---

## Testing Examples

```bash
# Test relationship queries
.venv\Scripts\python.exe tools\test_retrieval.py "people carrying bags"

# Test location filtering
.venv\Scripts\python.exe tools\test_retrieval.py "activity in parking lot"

# Test object search
.venv\Scripts\python.exe tools\test_retrieval.py "backpack in lobby"

# Test negative/no-match
.venv\Scripts\python.exe tools\test_retrieval.py "yellow helicopter on roof"

# Get JSON output for backend integration
.venv\Scripts\python.exe tools\test_retrieval.py "red handbag" --json
```

---

## Integration with Member 3 (LLM Backend)

Member 3 calls the search API:

```python
from events.event_search import search_events

# Member 3 receives user query from chatbot
user_query = "Did anyone carry a red bag near the gate?"

# Query Member 2's retrieval system
results = search_events(user_query, top_k=3)

# Member 3 uses the evidence to generate natural language response
for event in results.get("events", []):
    event_id = event["event_id"]
    location = event["location"]
    time = event["time_start"]
    clip_path = event["evidence"]["clip_path"]
    # Feed to LLM for response generation
```

---

## Evaluation Metrics (PS05 Requirement)

Run the full evaluation suite:

```bash
.venv\Scripts\python.exe evaluation\metrics.py
```

**Metrics Computed:**
- **Recall@1, Recall@5, Recall@10**: Percentage of relevant events retrieved in top-K
- **MRR (Mean Reciprocal Rank)**: Average rank position of first relevant result
- **Camera Accuracy**: Percentage of queries returning correct camera ID
- **Latency (P50, P95)**: Query response time percentiles

---

## Dependencies

```
torch>=2.0.0              # PyTorch for SigLIP model
transformers>=4.40.0      # HuggingFace transformers
chromadb>=0.5.0           # Vector database
numpy>=1.26.0             # Array operations
pandas>=2.0.0             # Data manipulation
scikit-learn>=1.3.0       # Evaluation metrics
tqdm>=4.66.0              # Progress bars
```

---

## Member 1 Integration Checklist

- [ ] Member 1 provides CVEvent JSON/JSONL output file
- [ ] Verify JSON schema matches expected format (event_id, camera, time, entities, relationship, evidence)
- [ ] Run `ingest_events_from_file()` to index events
- [ ] Test retrieval with sample queries
- [ ] Validate evidence paths (frame_path, clip_path) point to existing files
- [ ] Confirm camera IDs in config.py match Member 1's camera IDs

---

## Notes

- **No video file embedding**: Member 2 does NOT process raw .mp4 video files. All video analysis is done by Member 1.
- **Metadata-driven**: Search is powered by text embeddings of structured event descriptions.
- **Persistent storage**: ChromaDB data persists in `chroma_db/` folder across sessions.
- **GPU optional**: SigLIP text encoding runs efficiently on CPU. GPU accelerates but is not required.

---

## Contact

For questions about Member 2's retrieval pipeline, refer to this README or check the inline code documentation in `events/` and `retrieval/` modules.
