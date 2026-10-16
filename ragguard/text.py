"""Shared tokenization used by retrieval, the offline LLM and the metrics."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")

STOPWORDS = frozenset(
    """a an and are as at be been before by can do does for from has have how i if in
    into is it its may must my of on or our should so than that the their them then
    there these they this to was we what when where which who why will with within
    you your after any each every per not no only also must""".split()
)


def stem(token: str) -> str:
    """Very small suffix stripper so that 'bolts'/'bolt' and 'tested'/'test' match."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, split, drop stopwords and stem."""
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
