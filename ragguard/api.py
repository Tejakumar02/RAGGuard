"""FastAPI service: POST /ask and GET /health."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from ragguard import __version__
from ragguard.pipeline import RAGPipeline

DEFAULT_DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"

app = FastAPI(
    title="RAGGuard",
    version=__version__,
    description="Question answering over plant documents, with cited sources.",
)


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    docs = os.environ.get("RAGGUARD_DOCS", str(DEFAULT_DOCS))
    return RAGPipeline.from_directory(docs)


PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=4, ge=1, le=10)


class Source(BaseModel):
    n: int
    id: str
    source: str
    title: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    llm: str
    citations: list[int]
    sources: list[Source]


@app.get("/health")
def health(pipeline: PipelineDep) -> dict:
    return {"status": "ok", "chunks": len(pipeline.chunks), "llm": pipeline.llm.name}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, pipeline: PipelineDep) -> AskResponse:
    answer = pipeline.ask(request.question, top_k=request.top_k)
    return AskResponse(
        answer=answer.text,
        llm=answer.llm,
        citations=answer.citations,
        sources=[
            Source(n=n, id=c.id, source=c.source, title=c.title, score=s, text=c.text)
            for n, (c, s) in enumerate(zip(answer.contexts, answer.scores, strict=True), 1)
        ],
    )
