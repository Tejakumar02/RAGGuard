import pytest

from ragguard.ingest import chunk_text, load_corpus, split_sections
from tests.conftest import DOCS


def test_split_sections_reads_title_and_headings():
    markdown = "# Handbook\n\n## Safety\nWear gloves.\n\n## Tools\nUse a wrench."
    title, sections = split_sections(markdown)
    assert title == "Handbook"
    assert sections == [("Safety", "Wear gloves."), ("Tools", "Use a wrench.")]


def test_chunk_text_respects_limit_and_overlaps():
    text = " ".join(f"Sentence number {i} has five words." for i in range(20))
    chunks = chunk_text(text, max_words=20, overlap_words=6)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 20 for c in chunks)
    # last sentence of a chunk is repeated at the start of the next one
    assert chunks[1].startswith(chunks[0].split(". ")[-1].rstrip("."))


def test_chunk_text_never_splits_inside_a_sentence():
    text = "First sentence here. Second sentence here. Third sentence here."
    for chunk in chunk_text(text, max_words=4, overlap_words=0):
        assert chunk.endswith(".")


@pytest.mark.parametrize("max_words,overlap", [(0, 0), (10, 10), (10, -1)])
def test_chunk_text_rejects_bad_settings(max_words, overlap):
    with pytest.raises(ValueError):
        chunk_text("Some text.", max_words=max_words, overlap_words=overlap)


def test_load_corpus_builds_unique_ids_with_titles():
    chunks = load_corpus(DOCS)
    assert len({c.id for c in chunks}) == len(chunks)
    assert {c.source for c in chunks} >= {"lockout_tagout.md", "torque_and_fastening.md"}
    # every chunk is titled with its document; section chunks also carry the heading
    torque = [c for c in chunks if c.source == "torque_and_fastening.md"]
    assert all(c.title.startswith("Torque and Fastening Standards") for c in torque)
    assert "Torque and Fastening Standards > Torque values" in {c.title for c in torque}


def test_load_corpus_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")


def test_load_corpus_empty_directory(tmp_path):
    with pytest.raises(ValueError):
        load_corpus(tmp_path)
