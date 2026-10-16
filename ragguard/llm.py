"""Answer generators: Groq (real LLM) and an offline extractive baseline for CI."""

from __future__ import annotations

import os
from typing import Protocol

from ragguard.ingest import Chunk, split_sentences
from ragguard.text import tokenize

NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = (
    "You answer questions about internal plant documents. "
    "Use ONLY the numbered context passages. "
    "After every sentence, cite the passage(s) it came from like [1] or [2][3]. "
    f'If the context does not contain the answer, reply exactly: "{NO_ANSWER}" '
    "Be concise: at most three sentences."
)


def build_user_prompt(question: str, contexts: list[Chunk]) -> str:
    blocks = "\n\n".join(f"[{n}] ({c.title})\n{c.text}" for n, c in enumerate(contexts, 1))
    return f"Context passages:\n\n{blocks}\n\nQuestion: {question}\nAnswer:"


class LLM(Protocol):
    name: str

    def generate(self, question: str, contexts: list[Chunk]) -> str: ...


class GroqLLM:
    """Calls a Groq-hosted model. Needs GROQ_API_KEY."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from groq import Groq  # imported lazily so offline use needs no SDK

        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self.model = model or os.environ.get("RAGGUARD_GROQ_MODEL", "llama-3.3-70b-versatile")
        self.name = f"groq:{self.model}"
        self._client = Groq(api_key=key)

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return NO_ANSWER
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=300,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, contexts)},
            ],
        )
        return (response.choices[0].message.content or "").strip()


class ExtractiveLLM:
    """Deterministic, offline baseline: returns the best-matching context sentences.

    It needs no network or API key, so the evaluation gate can run on every pull
    request and any change in the score comes from retrieval/chunking, not LLM noise.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 2, min_coverage: float = 0.5) -> None:
        self.max_sentences = max_sentences
        self.min_coverage = min_coverage

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        q_terms = set(tokenize(question))
        if not q_terms or not contexts:
            return NO_ANSWER
        candidates = []
        for n, chunk in enumerate(contexts, 1):
            for position, sentence in enumerate(split_sentences(chunk.text)):
                overlap = len(q_terms & set(tokenize(sentence)))
                if overlap:
                    candidates.append((overlap, -n, -position, sentence, n))
        if not candidates:
            return NO_ANSWER
        candidates.sort(reverse=True)
        if candidates[0][0] / len(q_terms) < self.min_coverage:
            return NO_ANSWER
        picked = candidates[: self.max_sentences]
        picked.sort(key=lambda c: (-c[1], -c[2]))  # keep document order
        return " ".join(f"{sentence} [{n}]" for *_, sentence, n in picked)


def get_llm(name: str | None = None) -> LLM:
    """Pick a generator from `name` or RAGGUARD_LLM ("groq" | "extractive").

    Default: Groq when GROQ_API_KEY is set, otherwise the offline extractive baseline.
    """
    choice = (name or os.environ.get("RAGGUARD_LLM") or "").lower()
    if not choice:
        choice = "groq" if os.environ.get("GROQ_API_KEY") else "extractive"
    if choice == "groq":
        return GroqLLM()
    if choice == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"Unknown LLM '{choice}', expected 'groq' or 'extractive'")
