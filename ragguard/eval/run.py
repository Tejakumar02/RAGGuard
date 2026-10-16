"""Run the golden-set evaluation and fail (exit 1) if quality drops below thresholds.

Usage:
    python -m ragguard.eval.run                      # offline extractive baseline
    python -m ragguard.eval.run --llm groq           # real LLM (needs GROQ_API_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ragguard.eval import metrics
from ragguard.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[2]

METRIC_LABELS = {
    "hit_at_k": "Retrieval hit@k",
    "mrr": "Retrieval MRR",
    "faithfulness": "Faithfulness",
    "citation_validity": "Citation validity",
    "keyword_recall": "Answer keyword recall",
    "abstention_accuracy": "Abstention accuracy",
}


@dataclass
class CaseResult:
    id: str
    question: str
    answerable: bool
    answer: str
    sources: list[str]
    scores: dict[str, float] = field(default_factory=dict)


def load_golden(path: Path) -> list[dict]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        for key in ("id", "question"):
            if key not in case:
                raise ValueError(f"{path}:{line_no} is missing '{key}'")
        case.setdefault("answerable", True)
        case.setdefault("expected_sources", [])
        case.setdefault("keywords", [])
        cases.append(case)
    if not cases:
        raise ValueError(f"No cases in {path}")
    return cases


def evaluate(pipeline: RAGPipeline, cases: list[dict], k: int) -> list[CaseResult]:
    results = []
    for case in cases:
        answer = pipeline.ask(case["question"], top_k=k)
        result = CaseResult(
            id=case["id"],
            question=case["question"],
            answerable=case["answerable"],
            answer=answer.text,
            sources=answer.sources,
        )
        abstained = metrics.is_abstention(answer.text)
        if case["answerable"]:
            expected = case["expected_sources"]
            # an abstention on an answerable question scores 0 on the answer metrics
            faithful = metrics.faithfulness(answer.text, answer.contexts)
            cited = metrics.citation_validity(answer.citations, len(answer.contexts))
            result.scores = {
                "hit_at_k": metrics.hit_at_k(answer.sources, expected, k),
                "mrr": metrics.reciprocal_rank(answer.sources, expected),
                "faithfulness": 0.0 if abstained else faithful,
                "citation_validity": 0.0 if abstained else cited,
                "keyword_recall": metrics.keyword_recall(answer.text, case["keywords"]),
            }
        else:
            result.scores = {"abstention_accuracy": float(abstained)}
        results.append(result)
    return results


def aggregate(results: list[CaseResult]) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for r in results:
        for name, value in r.scores.items():
            totals.setdefault(name, []).append(value)
    return {name: round(sum(v) / len(v), 4) for name, v in totals.items()}


def check(summary: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    failures = []
    for name, minimum in thresholds.items():
        value = summary.get(name)
        if value is None:
            failures.append(f"{name}: metric missing (no cases measured it)")
        elif value < minimum:
            failures.append(f"{name}: {value:.3f} < required {minimum:.3f}")
    return failures


def render_markdown(meta: dict, summary: dict, thresholds: dict, results: list[CaseResult]) -> str:
    lines = [
        "# RAGGuard evaluation report",
        "",
        f"- **Generator:** `{meta['llm']}`  ",
        f"- **Cases:** {meta['cases']} ({meta['answerable']} answerable, "
        f"{meta['cases'] - meta['answerable']} unanswerable)  ",
        f"- **top_k:** {meta['top_k']}  ",
        f"- **Result:** {'✅ PASSED' if meta['passed'] else '❌ FAILED'}",
        "",
        "| Metric | Score | Threshold | Status |",
        "|---|---:|---:|:---:|",
    ]
    for name, label in METRIC_LABELS.items():
        if name not in summary:
            continue
        minimum = thresholds.get(name)
        status = "—" if minimum is None else ("✅" if summary[name] >= minimum else "❌")
        shown_min = "—" if minimum is None else f"{minimum:.2f}"
        lines.append(f"| {label} | {summary[name]:.3f} | {shown_min} | {status} |")

    misses = [r for r in results if any(v < 1.0 for v in r.scores.values())]
    lines += ["", f"## Cases below 1.0 ({len(misses)})", ""]
    if misses:
        lines += ["| Case | Question | Weak metrics |", "|---|---|---|"]
        for r in misses:
            weak = ", ".join(f"{k}={v:.2f}" for k, v in r.scores.items() if v < 1.0)
            lines.append(f"| {r.id} | {r.question} | {weak} |")
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAGGuard against a golden set.")
    parser.add_argument("--docs", type=Path, default=ROOT / "data" / "docs")
    parser.add_argument("--golden", type=Path, default=ROOT / "eval" / "golden.jsonl")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "eval" / "thresholds.toml")
    parser.add_argument("--llm", default="extractive", help="'extractive' (offline) or 'groq'")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args(argv)

    with args.thresholds.open("rb") as f:
        config = tomllib.load(f)
    thresholds = config.get(args.llm, config.get("default", {}))

    pipeline = RAGPipeline.from_directory(args.docs, llm=args.llm, top_k=args.top_k)
    cases = load_golden(args.golden)

    started = time.perf_counter()
    results = evaluate(pipeline, cases, args.top_k)
    elapsed = time.perf_counter() - started

    summary = aggregate(results)
    failures = check(summary, thresholds)
    meta = {
        "llm": pipeline.llm.name,
        "cases": len(results),
        "answerable": sum(r.answerable for r in results),
        "top_k": args.top_k,
        "seconds": round(elapsed, 2),
        "passed": not failures,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "meta": meta,
        "summary": summary,
        "thresholds": thresholds,
        "failures": failures,
        "cases": [r.__dict__ for r in results],
    }
    (args.out / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = render_markdown(meta, summary, thresholds, results)
    (args.out / "eval_report.md").write_text(markdown, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(markdown)

    print(markdown)
    if failures:
        print("Quality gate FAILED:", *failures, sep="\n  - ")
        return 1
    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
