# -*- coding: utf-8 -*-
"""ROUGE metric computation for RAG evaluation."""
from typing import Dict

from rouge_score import rouge_scorer


def compute_rouge(reference: str, prediction: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        reference: Ground-truth answer.
        prediction: Model-generated answer.

    Returns:
        Dict with keys: rouge_1, rouge_2, rouge_l (each an F1 float in [0,1]).
    """
    if not reference or not prediction:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}

    reference = reference.strip()
    prediction = prediction.strip()

    try:
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=False
        )
        scores = scorer.score(reference, prediction)
        return {
            "rouge_1": scores["rouge1"].fmeasure,
            "rouge_2": scores["rouge2"].fmeasure,
            "rouge_l": scores["rougeL"].fmeasure,
        }
    except Exception:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
