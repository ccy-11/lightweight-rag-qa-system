# -*- coding: utf-8 -*-
"""all-MiniLM-L6-v2 embedding model (lightweight English)."""
from typing import List

from sentence_transformers import SentenceTransformer

from src.embedding.base import EmbeddingModel


class AllMiniLML6V2(EmbeddingModel):
    model_name = "all-minilm-l6-v2"
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    dim = 384

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
            batch_size=64,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vectors]
