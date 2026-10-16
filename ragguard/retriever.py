"""Okapi BM25 retriever over document chunks (no external services needed)."""

from __future__ import annotations

import math
from collections import Counter

from ragguard.ingest import Chunk
from ragguard.text import tokenize


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        # The section title is indexed too, so headings help retrieval.
        self._tf = [Counter(tokenize(f"{c.title} {c.text}")) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = sum(self._len) / len(self._len)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query: str) -> list[float]:
        terms = tokenize(query)
        scores = []
        for tf, length in zip(self._tf, self._len, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self._avg_len)
            s = 0.0
            for term in terms:
                freq = tf.get(term)
                if freq:
                    s += self._idf[term] * freq * (self.k1 + 1) / (freq + norm)
            scores.append(s)
        return scores

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return up to k chunks with a positive score, best first (ties keep corpus order)."""
        scores = self.score(query)
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [(self.chunks[i], scores[i]) for i in ranked[:k] if scores[i] > 0]
