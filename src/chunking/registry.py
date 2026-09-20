# -*- coding: utf-8 -*-
"""Chunk strategy registry - returns the correct strategy by name."""
from typing import Dict, Type

from src.chunking.base import ChunkStrategy
from src.chunking.fixed_overlap import FixedOverlapChunk
from src.chunking.semantic_paragraph import SemanticParagraphChunk

_STRATEGIES: Dict[str, Type[ChunkStrategy]] = {
    FixedOverlapChunk.name: FixedOverlapChunk,
    SemanticParagraphChunk.name: SemanticParagraphChunk,
}


def get_strategy(name: str) -> ChunkStrategy:
    """Return a ChunkStrategy instance by name.

    Raises:
        ValueError: if name is not a known strategy.
    """
    if name not in _STRATEGIES:
        available = ", ".join(sorted(_STRATEGIES.keys()))
        raise ValueError(f"Unknown chunk strategy: '{name}'. Available: {available}")
    return _STRATEGIES[name]()


def list_strategies() -> list:
    return sorted(_STRATEGIES.keys())
