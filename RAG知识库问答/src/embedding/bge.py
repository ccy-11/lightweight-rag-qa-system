# -*- coding: utf-8 -*-
"""BGE-small embedding model (Chinese-optimized)."""
from typing import List

from sentence_transformers import SentenceTransformer

from src.embedding.base import EmbeddingModel


class BgeSmall(EmbeddingModel):
    model_name = "bge-small"
    model_id = "BAAI/bge-small-zh-v1.5"
    dim = 512

    def __init__(self, device: str = "cpu", cache_dir: str = "./model_cache"):
        self.device = device
        self._model = SentenceTransformer(
            self.model_id,
            device=device,
            cache_folder=cache_dir,
        )
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vectors]
