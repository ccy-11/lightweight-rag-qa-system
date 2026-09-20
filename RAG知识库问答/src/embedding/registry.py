# -*- coding: utf-8 -*-
"""Embedding model registry."""
from typing import Dict, Type

from src.embedding.base import EmbeddingModel
from src.embedding.bge import BgeSmall
from src.embedding.minilm import AllMiniLML6V2

_MODELS: Dict[str, Type[EmbeddingModel]] = {
    BgeSmall.model_name: BgeSmall,
    AllMiniLML6V2.model_name: AllMiniLML6V2,
}


def get_embedding(name: str, device: str = "cpu", cache_dir: str = "./model_cache") -> EmbeddingModel:
    """Return an EmbeddingModel instance by name.

    Raises:
        ValueError: if name is not a known embedding model.
    """
    if name not in _MODELS:
        available = ", ".join(sorted(_MODELS.keys()))
        raise ValueError(f"Unknown embedding model: '{name}'. Available: {available}")
    return _MODELS[name](device=device, cache_dir=cache_dir)


def list_models() -> list:
    return sorted(_MODELS.keys())
