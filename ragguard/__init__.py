"""RAGGuard: a retrieval-augmented QA service with a built-in evaluation gate."""

from ragguard.ingest import Chunk, load_corpus
from ragguard.pipeline import Answer, RAGPipeline

__all__ = ["Answer", "Chunk", "RAGPipeline", "load_corpus"]
__version__ = "0.1.0"
