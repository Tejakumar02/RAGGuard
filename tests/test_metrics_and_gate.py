import json

from ragguard.eval import metrics
from ragguard.eval.run import aggregate, check, evaluate, load_golden, main
from tests.conftest import DOCS


def test_retrieval_metrics():
    retrieved = ["x.md", "a.md", "a.md"]
    assert metrics.hit_at_k(retrieved, ["a.md"], k=1) == 0.0
    assert metrics.hit_at_k(retrieved, ["a.md"], k=2) == 1.0
    assert metrics.reciprocal_rank(retrieved, ["a.md"]) == 0.5
    assert metrics.reciprocal_rank(retrieved, ["z.md"]) == 0.0


def test_faithfulness_flags_invented_facts(chunks):
    grounded = "Drain plugs are tightened to 30 Nm [1]."
    invented = "Drain plugs are welded shut by robots every night [1]."
    assert metrics.faithfulness(grounded, chunks) == 1.0
    assert metrics.faithfulness(invented, chunks) == 0.0


def test_faithfulness_checks_the_cited_passage(chunks):
    # true sentence, but it cites the wrong passage
    assert metrics.faithfulness("Drain plugs are tightened to 30 Nm [3].", chunks) == 0.0


def test_citation_validity():
    assert metrics.citation_validity([1, 2], n_contexts=3) == 1.0
    assert metrics.citation_validity([4], n_contexts=3) == 0.0
    assert metrics.citation_validity([], n_contexts=3) == 0.0


def test_keyword_recall_and_abstention():
    assert metrics.keyword_recall("It is 30 nm.", ["30 Nm", "cross"]) == 0.5
    assert metrics.is_abstention("I don’t know based on the provided documents.")
    assert not metrics.is_abstention("30 Nm [1]")


def test_check_reports_regressions_and_missing_metrics():
    failures = check({"mrr": 0.5}, {"mrr": 0.8, "hit_at_k": 0.9})
    assert len(failures) == 2


def test_golden_set_passes_baseline(pipeline):
    from pathlib import Path

    golden = Path(DOCS).parents[1] / "eval" / "golden.jsonl"
    summary = aggregate(evaluate(pipeline, load_golden(golden), k=4))
    assert summary["hit_at_k"] >= 0.95
    assert summary["abstention_accuracy"] == 1.0


def test_gate_fails_when_quality_drops(tmp_path):
    """A deliberately broken corpus (one document removed) must fail the gate."""
    docs = tmp_path / "docs"
    docs.mkdir()
    for path in DOCS.iterdir():
        if path.name != "torque_and_fastening.md":
            (docs / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    out = tmp_path / "reports"
    assert main(["--docs", str(docs), "--out", str(out)]) == 1
    report = json.loads((out / "eval_report.json").read_text(encoding="utf-8"))
    assert not report["meta"]["passed"]
    assert any(f.startswith("hit_at_k") for f in report["failures"])


def test_gate_passes_and_writes_reports(tmp_path):
    out = tmp_path / "reports"
    assert main(["--out", str(out)]) == 0
    assert (out / "eval_report.md").read_text(encoding="utf-8").startswith("# RAGGuard")
