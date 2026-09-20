# -*- coding: utf-8 -*-
"""QA pipeline: retrieval + prompt assembly + Qwen inference."""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.chunking.base import Chunk
from src.embedding.base import EmbeddingModel
from src.retrieval.prompt_template import build_prompt_with_chat_template
from src.retrieval.llm import QwenLLM
from src.vectorstore.faiss_store import FAISSStore

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    question: str
    answer: str
    context_chunks: List[Chunk] = field(default_factory=list)
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    chunk_ids: List[str] = field(default_factory=list)


class QAPipeline:
    """End-to-end RAG QA pipeline.

    Flow: question -> embed -> FAISS search -> top_k chunks -> prompt -> Qwen -> answer
    """

    def __init__(
        self,
        faiss_store: FAISSStore,
        embedding_model: EmbeddingModel,
        llm: QwenLLM,
        top_k: int = 5,
    ):
        self.faiss_store = faiss_store
        self.embedding_model = embedding_model
        self.llm = llm
        self.top_k = top_k

    def answer(self, question: str) -> Answer:
        """Answer a single question using the RAG pipeline."""
        # Step 1: Embed the question
        t0 = time.time()
        query_vec = self.embedding_model.embed([question])[0]

        # Step 2: Retrieve top-k chunks
        results = self.faiss_store.search(query_vec, top_k=self.top_k)
        chunk_ids = [cid for cid, _ in results]
        chunks = [
            self.faiss_store.get_chunk(cid) for cid in chunk_ids if cid
        ]
        retrieval_time = time.time() - t0

        # Step 3: Build prompt and generate
        t1 = time.time()
        prompt = build_prompt_with_chat_template(
            self.llm.tokenizer, question, chunks
        )
        raw_answer = self.llm.generate(prompt, max_tokens=512)
        generation_time = time.time() - t1

        return Answer(
            question=question,
            answer=raw_answer,
            context_chunks=chunks,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            chunk_ids=chunk_ids,
        )
