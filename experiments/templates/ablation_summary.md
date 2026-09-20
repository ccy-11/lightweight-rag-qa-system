# Ablation Summary

| Group | Chunking | Embedding | R-1 | R-2 | R-L | Avg Time (s) |
|-------|----------|-----------|-----|-----|-----|--------------|
| A | fixed_overlap | bge-small | {a1} | {a2} | {aL} | {at} |
| B | fixed_overlap | all-MiniLM | {b1} | {b2} | {bL} | {bt} |
| C | semantic_paragraph | bge-small | {c1} | {c2} | {cL} | {ct} |
| D | semantic_paragraph | all-MiniLM | {d1} | {d2} | {dL} | {dt} |

## Analysis
- **Chunking effect (A vs C, B vs D):** {split_analysis}
- **Embedding effect (A vs B, C vs D):** {embedding_analysis}
- **Latency tradeoff:** {latency_tradeoff}
- **Recommended config:** {recommended_config}
