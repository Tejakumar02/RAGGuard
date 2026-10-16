"""Retrieve -> generate -> cite."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ragguard.ingest import Chunk, load_corpus
from ragguard.llm import LLM, get_llm
from ragguard.retriever import BM25Retriever

_CITATION = re.compile(r"\[(\d+)\]")


def parse_citations(text: str) -> list[int]:
    """Return cited passage numbers in first-seen order, without duplicates."""
    seen: list[int] = []
    for match in _CITATION.finditer(text):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen


@dataclass
class Answer:
    question: str
    text: str
    contexts: list[Chunk]
    scores: list[float]
    citations: list[int]
    llm: str

    @property
    def sources(self) -> list[str]:
        return [c.source for c in self.contexts]


class RAGPipeline:
    def __init__(self, chunks: list[Chunk], llm: LLM, top_k: int = 4) -> None:
        self.chunks = chunks
        self.retriever = BM25Retriever(chunks)
        self.llm = llm
        self.top_k = top_k

    @classmethod
    def from_directory(
        cls,
        docs_dir: str | Path,
        llm: LLM | str | None = None,
        top_k: int = 4,
        max_words: int = 120,
        overlap_words: int = 30,
    ) -> RAGPipeline:
        chunks = load_corpus(docs_dir, max_words, overlap_words)
        if llm is None or isinstance(llm, str):
            llm = get_llm(llm)
        return cls(chunks, llm, top_k)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        hits = self.retriever.search(question, k=top_k or self.top_k)
        contexts = [chunk for chunk, _ in hits]
        text = self.llm.generate(question, contexts)
        return Answer(
            question=question,
            text=text,
            contexts=contexts,
            scores=[round(score, 4) for _, score in hits],
            citations=parse_citations(text),
            llm=self.llm.name,
        )
