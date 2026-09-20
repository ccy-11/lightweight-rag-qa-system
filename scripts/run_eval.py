# -*- coding: utf-8 -*-
"""Run the full ablation experiment matrix.

Usage:
    python scripts/run_eval.py --config config.yaml --seed 42
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.evaluation.run_experiment import load_config, run_all_experiments, print_summary_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiments/logs/run_eval.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", default=None, help="Override results output dir")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.out_dir:
        config["results_dir"] = args.out_dir

    logger.info(f"Seed: {args.seed}")
    results = run_all_experiments(config, seed=args.seed)
    print_summary_table(results)
    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
