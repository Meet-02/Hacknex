"""
[DEPRECATED / UNUSED]
embeddings/text_embedding.py
Convenience wrapper: query string(s) -> normalised embedding vector(s).
DISCONNECTED from active CV -> RAG pipeline.
The active pipeline does NOT use or generate embeddings.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Union, List
import numpy as np
from models.siglip_model import encode_text


def embed_query(query: Union[str, List[str]]) -> np.ndarray:
    """
    query: single string or list of strings.
    Returns numpy array (768,) for a single query or (N, 768) for a list.
    Already lowercased + padded inside encode_text.
    """
    single  = isinstance(query, str)
    queries = [query] if single else query
    embs    = encode_text(queries).numpy()
    return embs[0] if single else embs
