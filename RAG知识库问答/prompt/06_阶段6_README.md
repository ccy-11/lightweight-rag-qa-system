# 阶段6 Prompt：README.md（GitHub风格）

## 项目背景（上下文）

轻量级本地RAG知识库问答系统（GitHub开源 + ArXiv预印本 + 求职AI应用工程师）。

**技术栈（严格限定）：**
- Python 3.10+ / LangChain / BGE-small（默认）/ all-MiniLM-L6-v2（备选）/ FAISS / Qwen-1.8B-Chat（4bit，贪心解码）/ PyPDF2 / Rouge-1/2/L
- 运行：Colab 免费tier 或 16GB本地电脑
- 无Agent、无Web UI、无商业API

**消融实验：**
- 切分策略：fixed_overlap vs semantic_paragraph
- 嵌入模型：BGE-small vs all-MiniLM-L6-v2
- 指标：Rouge-1/2/L + 平均延迟

**目录结构（与阶段1一致）：**
src/（loading/、chunking/、embedding/、vectorstore/、retrieval/、evaluation/）、config.yaml、data/、scripts/、experiments/

---

## 本Prompt目标

请生成一份完整的 README.md，适合GitHub开源项目首页，按以下顺序：

### 1. 项目标题 + 一句话描述
- 标题：LocalRAG — 轻量本地RAG知识库问答系统
- 副标题：支持多切分策略 x 多嵌入模型消融实验，可本地运行，可Colab复现

### 2. 项目特色（Features）
- 纯本地运行，无商业API
- 两种切分策略可切换
- 两种嵌入模型可切换
- 4bit量化适配16GB内存
- 批量评测 + Rouge指标 + 实验矩阵

### 3. 架构示意图（Mermaid或ASCII）
- PDF -> 加载 -> 切分 -> 嵌入 -> FAISS -> 检索 -> Prompt组装 -> Qwen-1.8B -> 答案

### 4. 快速开始（Quick Start）
#### Colab（3步）
1. 克隆repo
2. 运行 %pip install -r requirements.txt
3. 运行 python scripts/build_index.py 然后 python scripts/ask.py

#### 本地（16GB）
- 环境准备（conda/venv）
- 下载模型（HuggingFace命令）
- 构建索引 + 问答

### 5. 配置说明
- config.yaml 关键字段表格（切分策略、嵌入模型、top_k、种子等）

### 6. 批量评测
- 如何运行：python scripts/run_eval.py --seed 42
- 输出文件说明（summary.csv、JSON快照、日志）

### 7. 实验结果（占位）
- 表格：4组实验的 Rouge-1/2/L + 平均延迟（数值留空，待填入）

### 8. 论文相关
- 指向 experiments/ 目录
- 说明消融实验设计思路（一段话）

### 9. License
- MIT

### 10. 致谢
- Qwen-1.8B-Chat（Qwen团队）、BGE-small（BAAI）
- LangChain、FAISS、SentenceTransformer

---

## 验收标准
- [ ] README 可直接复制到GitHub repo root
- [ ] 架构示意图清晰（ASCII或Mermaid均可）
- [ ] Colab 和本地两条路径都完整可操作
- [ ] 实验结果表格有占位（留待实际填入）
- [ ] 单个代码块不超过30行

## 边界（不要额外做）
- 不要写贡献指南、CHANGELOG
- 不要添加CI/CD说明
- 不要写论文正文
