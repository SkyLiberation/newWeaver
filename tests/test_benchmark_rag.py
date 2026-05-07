import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_rag.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_rag", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_case_passes_when_retrieval_and_answer_meet_thresholds():
    mod = _load_module()

    case = {
        "id": "rag_pass",
        "query": "What sets the API base URL?",
        "expected_sources": ["config.md"],
        "expected_context_keywords": ["API_BASE_URL"],
        "expected_answer_keywords": ["API_BASE_URL"],
    }
    results = [
        {
            "source": "config.md",
            "filename": "config.md",
            "content": "The API_BASE_URL setting must be set in .env.",
        }
    ]
    answer = "Set API_BASE_URL in your .env file."

    scored = mod._score_case(case, results=results, answer=answer)

    assert scored["passed"] is True
    assert scored["retrieval"]["source_recall"] == 1.0
    assert scored["retrieval"]["context_recall"] == 1.0
    assert scored["answer"]["keyword_recall"] == 1.0


def test_score_case_requires_abstention_for_unanswerable_queries():
    mod = _load_module()

    case = {
        "id": "rag_abstain",
        "query": "What is the CEO's private phone number?",
        "should_abstain": True,
    }

    scored = mod._score_case(case, results=[], answer="No relevant knowledge base documents were found.")

    assert scored["passed"] is True
    assert scored["answer"]["abstention_pass"] is True
