# -*- coding: utf-8 -*-
"""Semantic paragraph chunking strategy."""
from typing import List

from src.chunking.base import Chunk, ChunkStrategy
from src.loading.pdf_loader import PageText


class SemanticParagraphChunk(ChunkStrategy):
    """Chunk at paragraph boundaries, capping each chunk at max_chunk_size.

    Paragraphs are separated by blank lines or single newlines.
    Paragraphs are accumulated until adding the next paragraph would exceed
    max_chunk_size, at which point the accumulated text is emitted as a chunk.
    Deterministic: no randomness involved.
    """

    name = "semantic_paragraph"

    def chunk(self, pages: List[PageText], config: dict) -> List[Chunk]:
        max_chunk_size = config.get("max_chunk_size", 1024)
        strategy_cfg = config.get("semantic_paragraph", {})
        max_chunk_size = strategy_cfg.get("max_chunk_size", max_chunk_size)

        chunks: List[Chunk] = []
        idx = 0

        for page in pages:
            text = page.text
            doc_name = page.doc_name
            page_num = page.page_num

            # Split into paragraphs (blank line separated)
            paragraphs: List[str] = []
            current: List[str] = []
            for line in text.split("\n"):
                if line.strip():
                    current.append(line.strip())
                else:
                    if current:
                        paragraphs.append(" ".join(current))
                        current = []
            if current:
                paragraphs.append(" ".join(current))

            # Accumulate paragraphs into chunks
            buffer: List[str] = []
            buffer_len = 0
            for para in paragraphs:
                if buffer and buffer_len + len(para) + 1 > max_chunk_size:
                    chunk_text = " ".join(buffer).strip()
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
                    buffer = [para]
                    buffer_len = len(para)
                else:
                    buffer.append(para)
                    buffer_len += len(para) + 1

            if buffer:
                chunk_text = " ".join(buffer).strip()
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

        return chunks
