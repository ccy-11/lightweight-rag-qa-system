# -*- coding: utf-8 -*-
"""RAG prompt template for Qwen-1.8B-Chat."""
from typing import List

from src.chunking.base import Chunk

SYSTEM_PROMPT = (
    "你是一个知识库问答助手。请仅基于【上下文】中提供的内容回答问题。"
    "如果上下文中没有足够信息来回答问题，请明确说明'根据提供的资料，我无法回答该问题'。"
    "回答时请引用来源，格式为 (文档名:页码)。"
)

USER_TEMPLATE = """【上下文】
{context_blocks}

【问题】{question}
"""


def format_context_blocks(chunks: List[Chunk]) -> str:
    """Format retrieved chunks into context blocks with source annotations."""
    if not chunks:
        return "(无检索结果)"
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = f"{chunk.doc_name}:{chunk.page_num}"
        blocks.append(f"[{i}] ({source})\n{chunk.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, chunks: List[Chunk]) -> str:
    """Build the full prompt string for Qwen-1.8B-Chat.

    Uses Qwen chat template format.
    """
    user_content = USER_TEMPLATE.format(
        context_blocks=format_context_blocks(chunks),
        question=question,
    )
    # Qwen chat format
    prompt = (
        f"\n\n[system]\n{SYSTEM_PROMPT}"
        f"\n\n[user]\n{user_content}"
        f"\n\n[assistant]\n"
    )
    return prompt


def build_prompt_with_chat_template(
    tokenizer, question: str, chunks: List[Chunk]
) -> str:
    """Build prompt using the HuggingFace tokenizer chat template if available."""
    user_content = USER_TEMPLATE.format(
        context_blocks=format_context_blocks(chunks),
        question=question,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except Exception:
        return build_prompt(question, chunks)
