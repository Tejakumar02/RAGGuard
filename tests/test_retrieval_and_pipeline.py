import pytest

from ragguard.llm import NO_ANSWER, ExtractiveLLM, get_llm
from ragguard.pipeline import RAGPipeline, parse_citations
from ragguard.retriever import BM25Retriever
from ragguard.text import tokenize


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The bolts are tightened") == ["bolt", "tighten"]


def test_bm25_ranks_the_matching_chunk_first(chunks):
    hits = BM25Retriever(chunks).search("what torque for drain plugs", k=2)
    assert hits[0][0].source == "a.md"
    assert hits[0][1] > 0


def test_bm25_returns_nothing_for_unrelated_query(chunks):
    assert BM25Retriever(chunks).search("cafeteria menu") == []


def test_bm25_requires_chunks():
    with pytest.raises(ValueError):
        BM25Retriever([])


def test_parse_citations_dedupes_in_order():
    assert parse_citations("A [2]. B [1][2]. C [3]") == [2, 1, 3]


def test_extractive_llm_cites_its_source(chunks):
    answer = ExtractiveLLM().generate("What does a red tower lamp mean?", chunks)
    assert "skipped" in answer
    assert "[2]" in answer


def test_extractive_llm_abstains_without_support(chunks):
    assert ExtractiveLLM().generate("Who is the CEO?", chunks) == NO_ANSWER
    assert ExtractiveLLM().generate("anything", []) == NO_ANSWER


def test_get_llm_defaults_to_offline_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("RAGGUARD_LLM", raising=False)
    assert get_llm().name == "extractive"


def test_get_llm_rejects_unknown_provider():
    with pytest.raises(ValueError):
        get_llm("gpt-17")


def test_pipeline_answers_with_valid_citations(pipeline: RAGPipeline):
    answer = pipeline.ask("How often are torque wrenches calibrated?")
    assert "90 days" in answer.text
    assert answer.sources[0] == "torque_and_fastening.md"
    assert answer.citations and all(1 <= c <= len(answer.contexts) for c in answer.citations)


def test_pipeline_abstains_on_out_of_scope_question(pipeline: RAGPipeline):
    assert pipeline.ask("What is the Wi-Fi password for the guest network?").text == NO_ANSWER
