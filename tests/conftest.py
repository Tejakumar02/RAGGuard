from pathlib import Path

import pytest

from ragguard import RAGPipeline
from ragguard.ingest import Chunk
from ragguard.llm import ExtractiveLLM

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


@pytest.fixture(scope="session")
def pipeline() -> RAGPipeline:
    return RAGPipeline.from_directory(DOCS, llm=ExtractiveLLM())


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk("a.md#0", "a.md", "A > Torque", "Drain plugs are tightened to 30 Nm."),
        Chunk("b.md#0", "b.md", "B > Lamps", "A red tower lamp means a step was skipped."),
        Chunk("c.md#0", "c.md", "C > Shifts", "The second shift starts at 14:00."),
    ]
