"""Answer generation: turn retrieved chunks + a question into a prompt, call Claude."""
from __future__ import annotations

import os

from anthropic import Anthropic

from src.rag import RetrievedChunk

SYSTEM_PROMPT = (
    "Ты отвечаешь на вопросы только по тексту рассказа, который тебе дают "
    "в качестве контекста. Если ответа в контексте нет, честно скажи, что "
    "не знаешь. Отвечай коротко, на русском языке."
)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the context + question into a single user prompt.

    Kept separate from the API call so it can be unit-tested without
    hitting the network.
    """
    context = "\n\n".join(f"[Отрывок {i + 1}]\n{c.text}" for i, c in enumerate(chunks))
    return f"Контекст из рассказа:\n\n{context}\n\nВопрос: {question}"


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = build_prompt(question, chunks)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text
