# -*- coding: utf-8 -*-
"""Batch experiment runner: ablation over chunking strategy x embedding model."""
import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import yaml

from src.chunking.base import Chunk
from src.chunking.registry import get_strategy
from src.embedding.registry import get_embedding
from src.loading.pdf_loader import load_all_pdfs
from src.retrieval.llm import QwenLLM
from src.retrieval.qa_pipeline import QAPipeline
from src.vectorstore.faiss_store import FAISSStore
from src.evaluation.rouge import compute_rouge

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    split_strategy: str
    embedding_model: str
    rouge_1_f1: float
    rouge_2_f1: float
    rouge_l_f1: float
    avg_retrieval_time: float
    avg_generation_time: float
    total_wall_time: float
    num_questions: int
    raw_answers: List[Dict[str, Any]] = field(default_factory=list)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_index(
    config: dict,
    split_strategy: str,
    embedding_model: str,
) -> FAISSStore:
    """Build (or load) the FAISS index for the given strategy + embedding combo."""
    index_dir = config.get("index_dir", "./experiments/results")
    index_name = f"faiss_{embedding_model}_{split_strategy}.faiss"
    index_path = os.path.join(index_dir, index_name)

    pages = load_all_pdfs(config.get("pdf_dir", "./data/pdfs"))
    strategy = get_strategy(split_strategy)
    chunks = strategy.chunk(pages, config)
    logger.info(f"  {len(chunks)} chunks produced by {split_strategy}")

    emb = get_embedding(
        embedding_model,
        device=config.get("device", "cpu"),
        cache_dir=config.get("cache_dir", "./model_cache"),
    )
    logger.info(f"  Embedding {len(chunks)} chunks with {embedding_model} ...")
    vectors = emb.embed([c.text for c in chunks])

    store = FAISSStore(index_path=index_path, dim=emb.dim)
    store.add(chunks, vectors)
    return store, emb


def _load_qa_dataset(config: dict) -> List[Dict[str, str]]:
    """Load QA pairs from the configured dataset."""
    qa_dir = config.get("qa_dir", "./data/qa_dataset")
    qa_file = os.path.join(qa_dir, "qa_pairs.jsonl")
    pairs: List[Dict[str, str]] = []
    if os.path.exists(qa_file):
        with open(qa_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
    max_q = config.get("max_questions")
    if max_q:
        pairs = pairs[:max_q]
    logger.info(f"  {len(pairs)} QA pairs loaded")
    return pairs


def run_single_experiment(
    config: dict,
    split_strategy: str,
    embedding_model: str,
    seed: int = 42,
) -> ExperimentResult:
    """Run one full experiment: build index + answer all QA pairs + compute ROUGE."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"=== Experiment: {split_strategy} x {embedding_model} ===")
    t_start = time.time()

    store, emb = _build_index(config, split_strategy, embedding_model)
    qa_pairs = _load_qa_dataset(config)

    llm = QwenLLM(
        device=config.get("device", "cpu"),
        use_quantization=config.get("use_quantization", True),
        cache_dir=config.get("cache_dir", "./model_cache"),
    )

    pipeline = QAPipeline(
        faiss_store=store,
        embedding_model=emb,
        llm=llm,
        top_k=config.get("top_k", 5),
    )

    results: List[Dict[str, Any]] = []
    retrieval_times: List[float] = []
    generation_times: List[float] = []

    for i, pair in enumerate(qa_pairs):
        question = pair.get("question", "")
        reference = pair.get("reference", "")
        logger.info(f"  [{i+1}/{len(qa_pairs)}] {question[:60]}...")

        answer = pipeline.answer(question)
        rouge = compute_rouge(reference, answer.answer)

        results.append(
            {
                "question": question,
                "reference": reference,
                "prediction": answer.answer,
                "chunk_ids": answer.chunk_ids,
                "retrieval_time": answer.retrieval_time,
                "generation_time": answer.generation_time,
                "rouge_1": rouge["rouge_1"],
                "rouge_2": rouge["rouge_2"],
                "rouge_l": rouge["rouge_l"],
            }
        )
        retrieval_times.append(answer.retrieval_time)
        generation_times.append(answer.generation_time)

    total_time = time.time() - t_start
    n = max(1, len(results))

    exp = ExperimentResult(
        split_strategy=split_strategy,
        embedding_model=embedding_model,
        rouge_1_f1=sum(r["rouge_1"] for r in results) / n,
        rouge_2_f1=sum(r["rouge_2"] for r in results) / n,
        rouge_l_f1=sum(r["rouge_l"] for r in results) / n,
        avg_retrieval_time=sum(retrieval_times) / n,
        avg_generation_time=sum(generation_times) / n,
        total_wall_time=total_time,
        num_questions=len(results),
        raw_answers=results,
    )

    # Save per-experiment JSON
    out_dir = config.get("results_dir", "./experiments/results")
    os.makedirs(out_dir, exist_ok=True)
    exp_json_path = os.path.join(
        out_dir, f"{split_strategy}_{embedding_model}.json"
    )
    snapshot = {
        "split_strategy": split_strategy,
        "embedding_model": embedding_model,
        "seed": seed,
        "top_k": config.get("top_k", 5),
        "chunking_params": config.get("chunking", {}),
        "num_questions": exp.num_questions,
        "rouge_1_f1": exp.rouge_1_f1,
        "rouge_2_f1": exp.rouge_2_f1,
        "rouge_l_f1": exp.rouge_l_f1,
        "avg_retrieval_time": exp.avg_retrieval_time,
        "avg_generation_time": exp.avg_generation_time,
        "total_wall_time": exp.total_wall_time,
    }
    with open(exp_json_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    logger.info(f"  Saved snapshot to {exp_json_path}")

    # Save log
    log_dir = config.get("log_dir", "./experiments/logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{split_strategy}_{embedding_model}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "question": r["question"],
                        "reference": r["reference"],
                        "prediction": r["prediction"],
                        "rouge_1": r["rouge_1"],
                        "rouge_2": r["rouge_2"],
                        "rouge_l": r["rouge_l"],
                        "retrieval_time": r["retrieval_time"],
                        "generation_time": r["generation_time"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info(f"  Saved log to {log_path}")

    return exp


def run_all_experiments(
    config: dict,
    seed: int = 42,
) -> List[ExperimentResult]:
    """Run the full ablation matrix: 2 split strategies x 2 embedding models."""
    splits = ["fixed_overlap", "semantic_paragraph"]
    embeddings = ["bge-small", "all-minilm-l6-v2"]

    results: List[ExperimentResult] = []
    for split in splits:
        for emb in embeddings:
            exp = run_single_experiment(config, split, emb, seed=seed)
            results.append(exp)

    # Write summary CSV
    out_dir = config.get("results_dir", "./experiments/results")
    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split_strategy",
                "embedding_model",
                "rouge_1",
                "rouge_2",
                "rouge_l",
                "avg_retrieval_time",
                "avg_generation_time",
                "total_wall_time",
                "num_questions",
            ],
        )
        writer.writeheader()
        for exp in results:
            writer.writerow(
                {
                    "split_strategy": exp.split_strategy,
                    "embedding_model": exp.embedding_model,
                    "rouge_1": round(exp.rouge_1_f1, 4),
                    "rouge_2": round(exp.rouge_2_f1, 4),
                    "rouge_l": round(exp.rouge_l_f1, 4),
                    "avg_retrieval_time": round(exp.avg_retrieval_time, 3),
                    "avg_generation_time": round(exp.avg_generation_time, 3),
                    "total_wall_time": round(exp.total_wall_time, 1),
                    "num_questions": exp.num_questions,
                }
            )
    logger.info(f"Summary CSV written to {summary_path}")

    return results


def print_summary_table(results: List[ExperimentResult]) -> None:
    """Print a human-readable summary table."""
    print()
    print("=" * 90)
    print(f"{'Experiment':<35} {'R1':>6} {'R2':>6} {'RL':>6} {'Ret_s':>7} {'Gen_s':>7} {'Qs':>4}")
    print("-" * 90)
    for exp in results:
        label = f"{exp.split_strategy} x {exp.embedding_model}"
        print(
            f"{label:<35} "
            f"{exp.rouge_1_f1:6.3f} "
            f"{exp.rouge_2_f1:6.3f} "
            f"{exp.rouge_l_f1:6.3f} "
            f"{exp.avg_retrieval_time:7.2f} "
            f"{exp.avg_generation_time:7.2f} "
            f"{exp.num_questions:4d}"
        )
    print("=" * 90)
