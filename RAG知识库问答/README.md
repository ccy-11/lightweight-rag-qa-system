# LocalRAG — 轻量本地 RAG 知识库问答系统

> 支持多切分策略 × 多嵌入模型消融实验，可本地运行，可 Colab 复现。

## Features

- 纯本地运行，无商业 API 依赖
- 两种切分策略可切换（fixed_overlap / semantic_paragraph）
- 两种嵌入模型可切换（BGE-small / all-MiniLM-L6-v2）
- 4bit 量化适配 16GB 内存
- 批量评测 + ROUGE 指标 + 实验矩阵（2×2 消融）

## Architecture

```
PDF → 加载(PDFLoader) → 切分(ChunkStrategy) → 嵌入(EmbeddingModel)
                                        ↓
                 Qwen-1.8B-Chat(LLM) ← Prompt组装 ← FAISS检索
                                        ↓
                              Answer (Rouge评测)
```

## Quick Start

### Colab (3 steps)

1. Clone the repo
2. ```
   %pip install -r requirements.txt
   ```
3. ```
   python scripts/build_index.py
   python scripts/ask.py
   ```

### Local (16GB)

```bash
# Setup
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download models (one-time)
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen-1.8B-Chat')"

# Build index
python scripts/build_index.py --device cpu

# Ask questions
python scripts/ask.py --device cpu

# Run ablation experiments
python scripts/run_eval.py --seed 42
```

## Configuration

`config.yaml` key fields:

| Field | Default | Description |
|-------|---------|-------------|
| `device` | `cpu` | `cpu` or `cuda` |
| `embedding.model` | `bge-small` | Embedding model name |
| `chunking.strategy` | `fixed_overlap` | Chunking strategy |
| `top_k` | `5` | Retrieval top-k |
| `seed` | `42` | Random seed |
| `llm.use_quantization` | `true` | 4bit quantization |

## Evaluation

```bash
python scripts/run_eval.py --seed 42
```

Outputs:
- `experiments/results/summary.csv` — 4-group ablation summary
- `experiments/results/{split}_{embedding}.json` — per-experiment snapshot
- `experiments/logs/{split}_{embedding}.log` — per-question log

## Results (placeholder)

| Group | Chunking | Embedding | R-1 | R-2 | R-L | Time (s) |
|-------|----------|-----------|-----|-----|-----|----------|
| A | fixed_overlap | bge-small | - | - | - | - |
| B | fixed_overlap | all-MiniLM | - | - | - | - |
| C | semantic_paragraph | bge-small | - | - | - | - |
| D | semantic_paragraph | all-MiniLM | - | - | - | - |

## Paper

Ablation experiment design: 2 chunking strategies × 2 embedding models = 4 conditions.
See `experiments/templates/` for the experiment record and reproducibility checklist.

## License

MIT

## Credits

- Qwen-1.8B-Chat (Qwen team)
- BGE-small (BAAI)
- LangChain, FAISS, SentenceTransformer
