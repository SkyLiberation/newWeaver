"""Run a lightweight benchmark for RAG retrieval and answer quality."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_ABSTAIN_PATTERNS = (
    "not supported by the excerpts",
    "not supported by the provided excerpts",
    "no relevant knowledge base documents were found",
    "not found in the knowledge base",
    "cannot determine from the provided excerpts",
    "未在知识库中检索到",
    "根据提供的摘录无法确定",
    "提供的摘录不支持",
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"Benchmark line {line_no} must be a JSON object")
        query = str(data.get("query") or "").strip()
        if not query:
            raise ValueError(f"Benchmark line {line_no} is missing a non-empty query")
        cases.append(data)
    return cases


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _match_count(haystack: str, needles: Iterable[str]) -> int:
    normalized = _norm(haystack)
    return sum(1 for needle in needles if needle and _norm(needle) in normalized)


def _contains_abstention(text: str) -> bool:
    normalized = _norm(text)
    return any(pattern in normalized for pattern in _ABSTAIN_PATTERNS)


def _score_case(
    case: Dict[str, Any],
    *,
    results: List[Dict[str, Any]],
    answer: str,
) -> Dict[str, Any]:
    expected_sources = [str(item).strip() for item in case.get("expected_sources", []) if str(item).strip()]
    expected_context_keywords = [
        str(item).strip() for item in case.get("expected_context_keywords", []) if str(item).strip()
    ]
    expected_answer_keywords = [
        str(item).strip() for item in case.get("expected_answer_keywords", []) if str(item).strip()
    ]
    should_abstain = bool(case.get("should_abstain", False))

    retrieved_sources_blob = "\n".join(
        f"{item.get('source', '')}\n{item.get('filename', '')}" for item in results if isinstance(item, dict)
    )
    retrieved_context_blob = "\n".join(
        str(item.get("content", "")) for item in results if isinstance(item, dict)
    )

    source_hits = _match_count(retrieved_sources_blob, expected_sources)
    context_hits = _match_count(retrieved_context_blob, expected_context_keywords)
    answer_hits = _match_count(answer, expected_answer_keywords)

    source_recall = source_hits / max(1, len(expected_sources)) if expected_sources else None
    context_recall = context_hits / max(1, len(expected_context_keywords)) if expected_context_keywords else None
    answer_recall = answer_hits / max(1, len(expected_answer_keywords)) if expected_answer_keywords else None
    abstention_pass = _contains_abstention(answer) if should_abstain else None

    min_source_recall = float(case.get("min_source_recall", 1.0 if expected_sources else 0.0))
    min_context_recall = float(case.get("min_context_recall", 1.0 if expected_context_keywords else 0.0))
    min_answer_recall = float(case.get("min_answer_recall", 1.0 if expected_answer_keywords else 0.0))

    if should_abstain:
        passed = bool(abstention_pass)
    else:
        passed = True
        if source_recall is not None:
            passed = passed and source_recall >= min_source_recall
        if context_recall is not None:
            passed = passed and context_recall >= min_context_recall
        if answer_recall is not None:
            passed = passed and answer_recall >= min_answer_recall

    return {
        "id": str(case.get("id") or ""),
        "query": str(case.get("query") or ""),
        "passed": bool(passed),
        "should_abstain": should_abstain,
        "retrieval": {
            "top_k": len(results),
            "source_recall": source_recall,
            "context_recall": context_recall,
            "source_hits": source_hits,
            "context_hits": context_hits,
        },
        "answer": {
            "text": answer,
            "keyword_recall": answer_recall,
            "keyword_hits": answer_hits,
            "abstention_pass": abstention_pass,
        },
        "results": results,
    }


def _avg(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


async def _run_case(
    case: Dict[str, Any],
    *,
    model: str,
    n_results: int,
    principal_id: str,
) -> Dict[str, Any]:
    import main
    from starlette.requests import Request

    scope = {"type": "http", "method": "POST", "path": "/benchmark/rag", "headers": [], "state": {}}
    request = Request(scope)
    if principal_id:
        request.state.principal_id = principal_id

    results = await main.rag_manager.run_search(request, str(case.get("query") or ""), n_results=n_results)
    payload = await main.rag_manager.answer(
        request=request,
        query=str(case.get("query") or ""),
        model=model,
        n_results=n_results,
    )
    answer = str(payload.get("content") or "")
    return _score_case(case, results=results, answer=answer)


async def _run_benchmark(
    *,
    bench_file: Path,
    output: Path,
    model: str,
    n_results: int,
    max_cases: Optional[int],
    principal_id: str,
) -> Dict[str, Any]:
    import main

    if not getattr(main.settings, "rag_enabled", False):
        raise RuntimeError("RAG is not enabled. Set RAG_ENABLED=true before running the benchmark.")

    cases = _load_jsonl(bench_file)
    if max_cases is not None:
        cases = cases[: max(0, int(max_cases))]

    evaluated = []
    for case in cases:
        evaluated.append(
            await _run_case(
                case,
                model=model,
                n_results=n_results,
                principal_id=principal_id,
            )
        )

    summary = {
        "total_cases": len(evaluated),
        "passed_cases": sum(1 for item in evaluated if item.get("passed")),
        "pass_rate": (
            sum(1 for item in evaluated if item.get("passed")) / len(evaluated)
            if evaluated
            else 0.0
        ),
        "avg_source_recall": _avg(item["retrieval"].get("source_recall") for item in evaluated),
        "avg_context_recall": _avg(item["retrieval"].get("context_recall") for item in evaluated),
        "avg_answer_keyword_recall": _avg(item["answer"].get("keyword_recall") for item in evaluated),
        "abstention_accuracy": _avg(
            1.0 if item["answer"].get("abstention_pass") else 0.0
            for item in evaluated
            if item.get("should_abstain")
        ),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bench_file": str(bench_file),
        "model": model,
        "n_results": n_results,
        "principal_id": principal_id,
        "summary": summary,
        "cases": evaluated,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main_cli() -> int:
    parser = argparse.ArgumentParser(description="Benchmark RAG retrieval and answer quality.")
    parser.add_argument(
        "--bench-file",
        type=Path,
        default=ROOT / "eval" / "benchmarks" / "rag_sample_tasks.jsonl",
        help="Path to benchmark JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the JSON benchmark report.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model used by the RAG answer stage. Defaults to PRIMARY_MODEL when omitted.",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="How many chunks to retrieve per query.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on number of benchmark cases.",
    )
    parser.add_argument(
        "--principal-id",
        default="",
        help="Optional principal id for internal-auth isolated collections.",
    )
    args = parser.parse_args()

    model = str(args.model or "").strip()
    if not model:
        import main

        model = str(getattr(main.settings, "primary_model", "") or "").strip()

    report = asyncio.run(
        _run_benchmark(
            bench_file=args.bench_file,
            output=args.output,
            model=model,
            n_results=max(1, int(args.n_results)),
            max_cases=args.max_cases,
            principal_id=str(args.principal_id or "").strip(),
        )
    )
    print(
        json.dumps(
            {
                "total_cases": report["summary"]["total_cases"],
                "passed_cases": report["summary"]["passed_cases"],
                "pass_rate": report["summary"]["pass_rate"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
