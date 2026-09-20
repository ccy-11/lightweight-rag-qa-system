# Experiment Record

## Metadata
- **Experiment ID:** {exp_id}
- **Date:** {date}
- **Random seed:** {seed}
- **Hardware:** {hardware} (GPU model / CPU / RAM)
- **Software versions:** {versions} (torch, transformers, faiss-cpu, sentence-transformers, rouge-score)

## Parameters
- **Chunking strategy:** {split_strategy} (params: {split_params})
- **Embedding model:** {embedding_model} (model_id: {embedding_model_id})
- **top_k:** {top_k}
- **QA dataset:** {qa_dataset} (samples: {n_samples})

## Results

| Metric | Value |
|--------|-------|
| ROUGE-1 F1 | {rouge1} |
| ROUGE-2 F1 | {rouge2} |
| ROUGE-L F1 | {rougeL} |
| Avg retrieval time (s) | {avg_retrieval} |
| Avg generation time (s) | {avg_generation} |
| Total wall time (s) | {total_time} |

## Observations
- **Anomalies:** {anomalies}
- **Baseline comparison:** {baseline_comparison}
- **Conclusion:** {conclusion}
