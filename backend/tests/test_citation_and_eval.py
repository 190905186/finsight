from app.evaluation.runner import numeric_match, retrieval_recall_at_k, score_item
from app.graph.workflow import _citations_valid


def test_citation_resolution() -> None:
    citations = [
        {
            "source_id": "abc123",
            "document_name": "3M_2018_10K",
            "citation": "[3M_2018_10K | Cash Flow/p.60 | 2018]",
        }
    ]
    assert _citations_valid("Capex was $1577 [3M_2018_10K | Cash Flow/p.60 | 2018]", citations)
    assert not _citations_valid("Capex was $1577", [])


def test_retrieval_recall_and_numeric() -> None:
    citations = [
        {"document_name": "3M_2018_10K", "page_index": 59, "page_end": 59},
        {"document_name": "OTHER", "page_index": 1, "page_end": 1},
    ]
    required = [{"doc_name": "3M_2018_10K", "evidence_page_num": 59}]
    assert retrieval_recall_at_k(citations, required, k=5) == 1.0
    assert numeric_match("The filing reports $1,577.00 million.", [1577.0])


def test_right_number_without_page_is_not_pass() -> None:
    item = {
        "question_id": "q1",
        "expected_behavior": "answer",
        "required_citations": [{"doc_name": "3M_2018_10K", "evidence_page_num": 59}],
        "key_numbers": [1577.0],
    }
    response = {"status": "answered", "answer": "$1577.00", "citations": []}
    scored = score_item(item, response)
    assert scored["numeric_match"] is True
    assert scored["retrieval_recall_at_k"] == 0.0
    assert scored["pass"] is False
