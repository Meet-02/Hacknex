"""
[DEPRECATED / UNUSED]
models/siglip_model.py
SigLIP2 text encoder for CVEvent metadata embedding.
DISCONNECTED from active CV -> RAG pipeline.
The active pipeline does NOT use or generate embeddings.
Video embeddings will be provided by the user's future video embedding system.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from typing import List
import config

_MODEL     = None
_PROCESSOR = None
_DEVICE    = None


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_id: str = config.SIGLIP_MODEL_ID):
    """Load SigLIP2 text model and processor (cached after first call)."""
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None:
        return _MODEL, _PROCESSOR, _DEVICE
    _DEVICE    = get_device()
    _PROCESSOR = AutoProcessor.from_pretrained(model_id)
    _MODEL     = AutoModel.from_pretrained(model_id).to(_DEVICE).eval()
    print(f"[SigLIP2 Text] Loaded {model_id} on {_DEVICE}")
    return _MODEL, _PROCESSOR, _DEVICE


def encode_text(texts: List[str], batch_size: int = 32) -> torch.Tensor:
    """
    Encode text strings -> L2-normalised embeddings (N, 768).
    Lowercases and pads to config.TEXT_MAX_LENGTH tokens.
    """
    model, processor, device = load_model()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch  = [t.lower() for t in texts[i : i + batch_size]]
        inputs = processor(
            text=batch,
            padding=config.TEXT_PADDING,
            max_length=config.TEXT_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            embs = model.get_text_features(**inputs)
            if hasattr(embs, "pooler_output") and embs.pooler_output is not None:
                embs = embs.pooler_output
        embs = F.normalize(embs, p=2, dim=-1)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)   # (N, 768)
