# -*- coding: utf-8 -*-
"""Chunk strategy base classes and data structures."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from src.loading.pdf_loader import PageText


@dataclass
class Chunk:
    text: str
    page_num: int
    doc_name: str
    chunk_id: str
    strategy_name: str


class ChunkStrategy(ABC):
    """Abstract base for all chunking strategies."""

    name: str = "base"

    @abstractmethod
    def chunk(self, pages: List[PageText], config: dict) -> List[Chunk]:
        """Chunk a list of PageText objects and return Chunk list."""
        pass
