# -*- coding: utf-8 -*-
"""FAISS vector store with disk persistence."""
import json
import os
from typing import Dict, List, Tuple

import faiss
import numpy as np

from src.chunking.base import Chunk


class FAISSStore:
    """FAISS vector store bound to a specific embedding model.

    Index file is named faiss_{model_name}.faiss to prevent cross-model reuse.
    Metadata (chunk texts + id map) is stored in a sidecar .json file.
    """

    def __init__(self, index_path: str, dim: int):
        self.index_path = index_path
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: Dict[str, Chunk] = {}
        self._cid_to_pos: Dict[str, int] = {}
        self._pos_to_cid: Dict[int, str] = {}
        self._load()

    def _load(self) -> None:
        meta_path = self.index_path.replace(".faiss", ".json")
        if os.path.exists(self.index_path):
            self._index = faiss.read_index(self.index_path)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._cid_to_pos = {cid: int(pos) for cid, pos in meta.get("cid_to_pos", {}).items()}
            self._pos_to_cid = {int(pos): cid for cid, pos in self._cid_to_pos.items()}
            self._chunks = {
                cid: Chunk(**c) for cid, c in meta.get("chunks", {}).items()
            }
            print(f"Loaded FAISS index with {self._index.ntotal} vectors from {self.index_path}")
        else:
            print(f"WARNING: no metadata found for {self.index_path}")

    def add(self, chunks: List[Chunk], vectors: List[List[float]]) -> None:
        """Add chunks and their vectors to the index and persist to disk."""
        arr = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(arr)

        start_pos = self._index.ntotal
        self._index.add(arr)

        for i, chunk in enumerate(chunks):
            pos = start_pos + i
            self._cid_to_pos[chunk.chunk_id] = pos
            self._pos_to_cid[pos] = chunk.chunk_id
            self._chunks[chunk.chunk_id] = chunk

        self._save()
        print(f"FAISS index updated: {self._index.ntotal} vectors at {self.index_path}")

    def search(self, query_vec: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for top_k chunks.

        Returns:
            List of (chunk_id, score) tuples, sorted by descending similarity.
        """
        if self._index.ntotal == 0:
            return []
        arr = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(arr)
        scores, indices = self._index.search(arr, min(top_k, self._index.ntotal))
        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk_id = self._pos_to_cid.get(int(idx))
            if chunk_id:
                results.append((chunk_id, float(score)))
        return results

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self._chunks.get(chunk_id)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        faiss.write_index(self._index, self.index_path)
        meta_path = self.index_path.replace(".faiss", ".json")
        meta = {
            "cid_to_pos": self._cid_to_pos,
            "chunks": {
                cid: {
                    "text": c.text,
                    "page_num": c.page_num,
                    "doc_name": c.doc_name,
                    "chunk_id": c.chunk_id,
                    "strategy_name": c.strategy_name,
                }
                for cid, c in self._chunks.items()
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return self._index.ntotal

    def __bool__(self) -> bool:
        return self._index.ntotal > 0
