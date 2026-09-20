# -*- coding: utf-8 -*-
"""Embedding model base class."""
from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """Abstract base for embedding models."""

    model_name: str = "base"
    model_id: str = ""
    dim: int = 0

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts and return a list of vectors."""
        pass
