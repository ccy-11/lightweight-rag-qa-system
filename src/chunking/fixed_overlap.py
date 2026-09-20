# -*- coding: utf-8 -*-
"""Fixed-size overlapping chunking strategy."""
from typing import List

from src.chunking.base import Chunk, ChunkStrategy
from src.loading.pdf_loader import PageText


class FixedOverlapChunk(ChunkStrategy):
    """Chunk text at a fixed token count with overlap.

    Approximates tokens as 2 characters (conservative for mixed CJK/Latin).
    Deterministic: no randomness involved.
    """

    name = "fixed_overlap"

    def chunk(self, pages: List[PageText], config: dict) -> List[Chunk]:
        chunk_size = config.get("chunk_size", 512)
        chunk_overlap = config.get("chunk_overlap", 64)
        strategy_cfg = config.get("fixed_overlap", {})
        chunk_size = strategy_cfg.get("chunk_size", chunk_size)
        chunk_overlap = strategy_cfg.get("chunk_overlap", chunk_overlap)

        step = max(1, chunk_size - chunk_overlap)
        chunks: List[Chunk] = []
        idx = 0

        for page in pages:
            text = page.text
            doc_name = page.doc_name
            page_num = page.page_num
            offset = 0
            while offset < len(text):
                end = min(offset + chunk_size, len(text))
                chunk_text = text[offset:end].strip()
                if chunk_text:
                    chunk_id = f"{doc_name}_p{page_num}_c{idx:04d}"
                    chunks.append(
                        Chunk(
                            text=chunk_text,
                            page_num=page_num,
                            doc_name=doc_name,
                            chunk_id=chunk_id,
                            strategy_name=self.name,
                        )
                    )
                    idx += 1
                offset += step
                if end >= len(text):
                    break

        return chunks
