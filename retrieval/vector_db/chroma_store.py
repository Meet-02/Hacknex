"""
vector_db/chroma_store.py
ChromaDB interface for CCTV evidence.

Key design decisions (from workflow doc):
- Collection uses cosine distance: hnsw:space = cosine  (default is L2!)
- Chroma returns *distance*; we return similarity = 1 - distance.
- Metadata values must be scalars (str/int/float/bool); no lists.
- Time filter example:
    where={"$and": [{"camera_id": {"$eq": "G336"}},
                     {"timestamp_sec": {"$gte": 50400}}]}
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Dict, Any, Optional
import chromadb
import config

_CLIENT     = None
_COLLECTION = None


def get_client() -> chromadb.PersistentClient:
    global _CLIENT
    if _CLIENT is None:
        os.makedirs(config.CHROMA_DB_PATH, exist_ok=True)
        _CLIENT = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    return _CLIENT


def get_collection() -> chromadb.Collection:
    global _COLLECTION
    if _COLLECTION is None:
        client = get_client()
        _COLLECTION = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},   # MUST set; default is L2
        )
    return _COLLECTION


def upsert(records: List[Dict[str, Any]]) -> None:
    """
    Upsert a list of records.
    Each record: {id, embedding (list|np.ndarray), metadata (scalar dict)}
    """
    collection = get_collection()
    ids = [r["id"] for r in records]
    embeddings = [
        r["embedding"].tolist() if isinstance(r["embedding"], np.ndarray)
        else list(r["embedding"])
        for r in records
    ]
    metadatas = [r["metadata"] for r in records]
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)


def query(
    embedding: np.ndarray,
    top_k: int = 50,
    where: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """
    Query the collection.
    Returns list sorted by similarity desc:
        [{id, metadata, similarity}, ...]
    """
    collection = get_collection()
    n = collection.count()
    if n == 0:
        return []

    kwargs: Dict[str, Any] = {
        "query_embeddings": [embedding.tolist()],
        "n_results": min(top_k, n),
        "include": ["metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    hits = []
    for doc_id, meta, dist in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "id":         doc_id,
            "metadata":   meta,
            "similarity": float(1.0 - dist),   # cosine: sim = 1 - distance
        })
    hits.sort(key=lambda h: h["similarity"], reverse=True)
    return hits


def count() -> int:
    return get_collection().count()


def delete_all() -> None:
    """Remove all records — useful for re-indexing in tests."""
    client = get_client()
    client.delete_collection(config.CHROMA_COLLECTION_NAME)
    global _COLLECTION
    _COLLECTION = None
