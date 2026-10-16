"""Deterministic RAG metrics. No LLM judge is needed, so results are reproducible."""

from __future__ import annotations

import re

from ragguard.ingest import Chunk, split_sentences
from ragguard.llm import NO_ANSWER
from ragguard.text import tokenize

_CITATION = re.compile(r"\[\d+\]")
_ABSTAIN_PATTERNS = (
    NO_ANSWER.lower(),
    "i don't know",
    "i do not know",
    "not in the provided",
    "does not contain",
)


def hit_at_k(retrieved_sources: list[str], expected_sources: list[str], k: int) -> float:
    """1.0 if any expected document appears in the top-k retrieved chunks."""
    return float(any(s in expected_sources for s in retrieved_sources[:k]))


def reciprocal_rank(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    for rank, source in enumerate(retrieved_sources, 1):
        if source in expected_sources:
            return 1.0 / rank
    return 0.0


def is_abstention(answer: str) -> bool:
    text = answer.lower().replace("’", "'")
    return any(p in text for p in _ABSTAIN_PATTERNS)


def citation_validity(citations: list[int], n_contexts: int) -> float:
    """1.0 if the answer cites at least one passage and every citation exists."""
    if not citations:
        return 0.0
    return float(all(1 <= c <= n_contexts for c in citations))


def faithfulness(answer: str, contexts: list[Chunk], support_threshold: float = 0.7) -> float:
    """Share of answer sentences whose content words appear in the cited passages.

    A sentence is "supported" when at least `support_threshold` of its content
    tokens occur in the passages it cites (or in all retrieved passages if it cites
    none). This is a lexical proxy for hallucination: it catches invented numbers
    and facts, but it is not a full entailment check.
    """
    sentences = [s for s in split_sentences(answer) if tokenize(_CITATION.sub("", s))]
    if not sentences:
        return 0.0
    supported = 0
    for sentence in sentences:
        cited = [int(c) for c in re.findall(r"\[(\d+)\]", sentence)]
        pool = [contexts[c - 1] for c in cited if 1 <= c <= len(contexts)] or contexts
        pool_tokens = set(tokenize(" ".join(c.text for c in pool)))
        tokens = tokenize(_CITATION.sub("", sentence))
        if sum(t in pool_tokens for t in tokens) / len(tokens) >= support_threshold:
            supported += 1
    return supported / len(sentences)


def keyword_recall(answer: str, keywords: list[str]) -> float:
    """Share of the expected key facts (case-insensitive substrings) found in the answer."""
    if not keywords:
        return 1.0
    text = answer.lower()
    return sum(k.lower() in text for k in keywords) / len(keywords)
