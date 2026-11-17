from __future__ import annotations

import pytest

from app.services.rag_prompt_resolver import RAGPromptResolver


class FakePipeline:
    def __init__(self) -> None:
        self.call: tuple[str, int, dict[str, str] | None] | None = None

    def answer(self, query, *, top_k=5, filters=None):
        self.call = (query, top_k, filters)
        return {
            "answer": "FY2018 PP&E purchases were $1,577 million. [S1]",
            "citations": [{"source": "[S1]", "document": "3M_2018_10K", "page": 60}],
        }


def test_resolve_forwards_prompt_and_company_filter() -> None:
    pipeline = FakePipeline()
    resolver = RAGPromptResolver(pipeline)

    result = resolver.resolve(
        "  What were FY2018 PP&E purchases?  ",
        top_k=3,
        company=" 3M ",
    )

    assert pipeline.call == (
        "What were FY2018 PP&E purchases?",
        3,
        {"company": "3M"},
    )
    assert result["citations"][0]["page"] == 60


def test_resolve_uses_shared_corpus_when_company_is_omitted() -> None:
    pipeline = FakePipeline()

    RAGPromptResolver(pipeline).resolve("What were FY2018 PP&E purchases?")

    assert pipeline.call == (
        "What were FY2018 PP&E purchases?",
        5,
        None,
    )


def test_resolve_can_scope_to_a_specific_report() -> None:
    pipeline = FakePipeline()

    RAGPromptResolver(pipeline).resolve(
        "What were FY2018 PP&E purchases?",
        company="3M",
        doc_name="3M_2018_10K",
    )

    assert pipeline.call == (
        "What were FY2018 PP&E purchases?",
        5,
        {"company": "3M", "doc_name": "3M_2018_10K"},
    )


@pytest.mark.parametrize(
    ("prompt", "company", "message"),
    [
        ("  ", None, "prompt must not be empty"),
        ("What were PP&E purchases?", "  ", "company must not be empty"),
    ],
)
def test_resolve_rejects_blank_prompt_and_company(
    prompt: str, company: str | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RAGPromptResolver(FakePipeline()).resolve(prompt, company=company)


def test_resolve_rejects_blank_document_name() -> None:
    with pytest.raises(ValueError, match="doc_name must not be empty"):
        RAGPromptResolver(FakePipeline()).resolve(
            "What were FY2018 PP&E purchases?",
            doc_name=" ",
        )
