# -*- coding: utf-8 -*-
"""Build the FAISS index from PDF documents.

Usage:
    python scripts/build_index.py --config config.yaml
"""
import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.chunking.registry import get_strategy
from src.embedding.registry import get_embedding
from src.loading.pdf_loader import load_all_pdfs
from src.vectorstore.faiss_store import FAISSStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from PDFs")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--strategy", default=None, help="Override chunking strategy")
    parser.add_argument("--embedding", default=None, help="Override embedding model")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    strategy_name = args.strategy or config.get("chunking", {}).get("strategy", "fixed_overlap")
    emb_name = args.embedding or config.get("embedding", {}).get("model", "bge-small")
    device = args.device or config.get("device", "cpu")
    cache_dir = config.get("cache_dir", "./model_cache")
    index_dir = config.get("index_dir", "./experiments/results")
    pdf_dir = config.get("pdf_dir", "./data/pdfs")

    index_path = os.path.join(index_dir, f"faiss_{emb_name}_{strategy_name}.faiss")

    logger.info(f"Strategy: {strategy_name} | Embedding: {emb_name} | Device: {device}")

    # Load PDFs
    logger.info(f"Loading PDFs from {pdf_dir} ...")
    pages = load_all_pdfs(pdf_dir)

    # Chunk
    logger.info("Chunking ...")
    strategy = get_strategy(strategy_name)
    chunks = strategy.chunk(pages, config)
    logger.info(f"  {len(chunks)} chunks produced")

    # Embed
    logger.info(f"Embedding with {emb_name} ...")
    emb = get_embedding(emb_name, device=device, cache_dir=cache_dir)
    vectors = emb.embed([c.text for c in chunks])

    # Store
    logger.info(f"Building FAISS index at {index_path} ...")
    os.makedirs(index_dir, exist_ok=True)
    store = FAISSStore(index_path=index_path, dim=emb.dim)
    store.add(chunks, vectors)

    logger.info("Done.")


if __name__ == "__main__":
    main()
