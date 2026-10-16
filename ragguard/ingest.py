"""Load documents and split them into section-aware, overlapping chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a document."""

    id: str  # e.g. "lockout_tagout.md#2"
    source: str  # file name, e.g. "lockout_tagout.md"
    title: str  # "Document title > Section heading"
    text: str


def read_document(path: Path) -> str:
    """Return the plain text of a .md, .txt or .pdf file."""
    if path.suffix.lower() == ".pdf":
        import fitz  # PyMuPDF is optional and only needed for PDFs

        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    return path.read_text(encoding="utf-8")


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    return [s for s in _SENTENCE_END.split(text) if s]


def split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split markdown into (document title, [(section heading, section body)])."""
    doc_title = ""
    sections: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            level, heading = len(match.group(1)), match.group(2)
            if level == 1 and not doc_title:
                doc_title = heading
                continue
            sections.append((heading, []))
        elif line.strip():
            if not sections:
                sections.append(("", []))
            sections[-1][1].append(line.strip())
    return doc_title, [(h, " ".join(body)) for h, body in sections if body]


def chunk_text(text: str, max_words: int = 120, overlap_words: int = 30) -> list[str]:
    """Pack whole sentences into chunks of at most ~max_words, with sentence overlap."""
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be in [0, max_words)")

    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and _word_count(current) + len(sentence.split()) > max_words:
            chunks.append(" ".join(current))
            # carry trailing sentences forward as overlap
            carry: list[str] = []
            for previous in reversed(current):
                if _word_count(carry) + len(previous.split()) > overlap_words:
                    break
                carry.insert(0, previous)
            current = carry
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _word_count(sentences: list[str]) -> int:
    return sum(len(s.split()) for s in sentences)


def load_corpus(
    directory: str | Path, max_words: int = 120, overlap_words: int = 30
) -> list[Chunk]:
    """Load every supported file in `directory` (sorted, non-recursive) into chunks."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    chunks: list[Chunk] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        doc_title, sections = split_sections(read_document(path))
        doc_title = doc_title or path.stem.replace("_", " ").title()
        index = 0
        for heading, body in sections:
            title = f"{doc_title} > {heading}" if heading else doc_title
            for piece in chunk_text(body, max_words, overlap_words):
                chunks.append(Chunk(f"{path.name}#{index}", path.name, title, piece))
                index += 1
    if not chunks:
        raise ValueError(f"No supported documents found in {directory}")
    return chunks
