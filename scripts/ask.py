# -*- coding: utf-8 -*-
"""Interactive QA CLI for the RAG pipeline.

Usage:
    python scripts/ask.py --config config.yaml
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.chunking.registry import get_strategy
from src.embedding.registry import get_embedding
from src.retrieval.llm import QwenLLM
from src.retrieval.qa_pipeline import QAPipeline
from src.vectorstore.faiss_store import FAISSStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RAG QA CLI")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--strategy", default=None, help="Override chunking strategy")
    parser.add_argument("--embedding", default=None, help="Override embedding model")
    parser.add_argument("--device", default=None, help="Override device")
    parser.add_argument("--question", default=None, help="Ask a single question and exit")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    strategy_name = args.strategy or config.get("chunking", {}).get("strategy", "fixed_overlap")
    emb_name = args.embedding or config.get("embedding", {}).get("model", "bge-small")
    device = args.device or config.get("device", "cpu")
    cache_dir = config.get("cache_dir", "./model_cache")
    index_dir = config.get("index_dir", "./experiments/results")
    top_k = config.get("top_k", 5)

    index_path = os.path.join(index_dir, f"faiss_{emb_name}_{strategy_name}.faiss")

    if not os.path.exists(index_path):
        logger.warning(
            f"Index not found at {index_path}. Run 'python scripts/build_index.py' first."
        )
        return

    logger.info(f"Loading index: {index_path}")
    store = FAISSStore(index_path=index_path, dim=config.get("embedding_dim", 512))
    emb = get_embedding(emb_name, device=device, cache_dir=cache_dir)
    llm = QwenLLM(device=device, cache_dir=cache_dir)
    pipeline = QAPipeline(store, emb, llm, top_k=top_k)

    if args.question:
        answer = pipeline.answer(args.question)
        print(f"\nQ: {answer.question}\nA: {answer.answer}")
        print(f"\nSources: {answer.chunk_ids}")
        return

    print("RAG QA - type a question (quit to exit)")
    while True:
        try:
            question = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        answer = pipeline.answer(question)
        print(f"\nA: {answer.answer}")
        if answer.chunk_ids:
            print(f"Sources: {', '.join(answer.chunk_ids)}")


if __name__ == "__main__":
    main()
