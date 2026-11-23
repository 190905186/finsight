from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.evaluation import load_safety_cases
from app.graph.workflow import (
    FinanceReportGraph,
    _as_facts,
    _requires_human_review_action,
)


class FakeRAGPipeline:
    def retrieve_many(
        self,
        queries: list[str],
        *,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        common = {
            "document": "Example_2024_10K",
            "page": 12,
            "metadata": {},
            "score": 0.1,
        }
        return [
            [
                {
                    **common,
                    "chunk_id": f"statement-{index}",
                    "snippet": f"Evidence for {query}",
                    "page_index": index,
                }
            ]
            for index, query in enumerate(queries)
        ]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "document": "Example_2024_10K",
                "page": 12,
                "snippet": "Revenue was $120 million, compared with $100 million in 2023.",
                "metadata": {},
                "source": "R1",
            }
        ]


class FakeLLMClient:
    def __enter__(self) -> FakeLLMClient:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        model: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        if "Extract only financial facts" in messages[0]["content"]:
            content = (
                '{"facts":{"revenue":{"value":"120","unit":"USD millions",'
                '"period":"FY2024","source":"R1",'
                '"quote":"Revenue was $120 million"},'
                '"prior_revenue":{"value":"100","unit":"USD millions",'
                '"period":"FY2023","source":"R1",'
                '"quote":"revenue of $100 million in 2023"}}}'
            )
        else:
            content = "Revenue increased based on the indexed filing [R1]."
        return {"choices": [{"message": {"content": content}}]}


def rate_limit_error() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://llm.example/chat")
    response = httpx.Response(429, request=request)
    return httpx.HTTPStatusError(
        "Too many requests", request=request, response=response
    )


def test_langgraph_returns_cited_report_and_threaded_memory() -> None:
    graph = FinanceReportGraph(
        rag_pipeline=FakeRAGPipeline(),
        llm_factory=FakeLLMClient,
    )

    first = graph.run(
        query="Summarize revenue growth",
        conversation_id="conversation-a",
        user_id="user-a",
    )
    second = graph.run(
        query="What is the source?",
        conversation_id="conversation-a",
        user_id="user-a",
    )
    other_thread = graph.run(
        query="Summarize revenue growth",
        conversation_id="conversation-b",
        user_id="user-a",
    )

    assert first["status"] == "ready"
    assert next(item for item in first["metrics"] if item["key"] == "revenue_growth")[
        "value"
    ] == "20.0000"
    assert first["citations"][0]["source"] == "R1"
    assert second["conversation_id"] == "conversation-a"
    assert other_thread["conversation_id"] == "conversation-b"


@pytest.mark.parametrize(
    "case",
    load_safety_cases(),
    ids=lambda case: case.case_id,
)
def test_consequential_actions_are_refused_before_tools(case) -> None:
    class NoCallRAGPipeline:
        def retrieve(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            raise AssertionError("A refused request must not retrieve evidence.")

        def retrieve_many(self, *args: Any, **kwargs: Any) -> list[list[dict[str, Any]]]:
            raise AssertionError("A refused request must not retrieve evidence.")

    class NoCallLLM:
        def __init__(self) -> None:
            raise AssertionError("A refused request must not call the model.")

    graph = FinanceReportGraph(
        rag_pipeline=NoCallRAGPipeline(),
        llm_factory=NoCallLLM,
    )
    result = graph.run(
        query=case.query,
        conversation_id=case.case_id,
        user_id="user-a",
    )

    assert result["status"] == "refused"
    assert result["answer"] == case.expected_answer
    assert result["citations"] == []
    assert result["agent_trace"] == ["safety_gate"]
    assert result["data_status"]["safety"] == "human_review_required"


def test_informational_finance_query_is_not_blocked_by_action_guard() -> None:
    assert not _requires_human_review_action(
        "What were 3M's FY2018 purchases of property, plant and equipment?"
    )


def test_report_research_targets_each_financial_statement() -> None:
    pipeline = FakeRAGPipeline()
    graph = FinanceReportGraph(
        rag_pipeline=pipeline,
        llm_factory=FakeLLMClient,
    )

    result = graph._retrieve(
        {
            "query": "Prepare report",
            "company": "3M",
            "fiscal_year": 2018,
            "report_mode": True,
            "retrieval_filters": {"doc_name": "3M_2018_10K"},
        }
    )

    assert len(result["evidence"]) == 3
    assert result["data_status"]["knowledge_base"] == "ready"


def test_report_synthesis_displays_only_calculated_metric_values() -> None:
    graph = FinanceReportGraph(
        rag_pipeline=FakeRAGPipeline(),
        llm_factory=FakeLLMClient,
    )

    result = graph._synthesize(
        {
            "query": "Generate a financial metrics report",
            "company": "Example",
            "fiscal_year": 2024,
            "report_mode": True,
            "metrics": [
                {
                    "key": "net_margin",
                    "label": "Net margin",
                    "value": "10.0000",
                    "unit": "%",
                    "status": "computed",
                    "formula": "net income / revenue",
                    "inputs": [
                        {"name": "Net income", "value": "12", "unit": "USD", "source": "R1"},
                        {"name": "Revenue", "value": "120", "unit": "USD", "source": "R1"},
                    ],
                    "source": "R1",
                    "source_urls": [],
                },
                {
                    "key": "gross_margin",
                    "label": "Gross margin",
                    "value": None,
                    "unit": "%",
                    "status": "unavailable",
                    "formula": "gross profit / revenue",
                    "inputs": [],
                    "source": "",
                    "source_urls": [],
                },
            ],
            "evidence": [{"source": "R1"}],
        }
    )

    assert "Net margin**: 10%" in result["answer"]
    assert "Gross margin**: Unavailable" in result["answer"]
    assert "[R1]" in result["answer"]


def test_citation_critic_performs_one_bounded_retry() -> None:
    class UncitedLLM(FakeLLMClient):
        calls = 0

        def create_chat_completion(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            type(self).calls += 1
            return {"choices": [{"message": {"content": "Revenue increased."}}]}

    graph = FinanceReportGraph(
        rag_pipeline=FakeRAGPipeline(),
        llm_factory=UncitedLLM,
    )

    result = graph.run(
        query="Summarize filing details",
        conversation_id="critic-retry",
        user_id="user-a",
    )

    assert UncitedLLM.calls == 2
    assert result["status"] == "ready"
    assert "did not pass citation verification" in result["answer"]
    assert "[R1]" in result["answer"]
    assert result["agent_trace"].count("financial_analyst") == 2
    assert result["agent_trace"].count("report_verifier") == 2


def test_product_chat_skips_financial_extraction_and_is_ready_with_citations() -> None:
    class NoCallLLM:
        def __init__(self) -> None:
            raise AssertionError("A product question should not invoke fact extraction.")

    graph = FinanceReportGraph(
        rag_pipeline=FakeRAGPipeline(),
        llm_factory=NoCallLLM,
    )
    evidence = [
        {
            "source": "R1",
            "document": "AMD_2022_10K",
            "page": 8,
            "snippet": "AMD offers processors, graphics, and adaptive computing products.",
            "metadata": {},
        }
    ]

    extracted = graph._extract(
        {
            "query": "What are AMD's major products and services as of FY22?",
            "report_mode": False,
            "evidence": evidence,
        }
    )
    verified = FinanceReportGraph._verify(
        {
            "answer": "AMD offers processors and graphics products [R1].",
            "evidence": evidence,
            "metrics": [],
            "report_mode": False,
        }
    )

    assert extracted["extraction_facts"] == {}
    assert extracted["data_status"]["fact_extraction"] == "not_requested"
    assert verified["status"] == "ready"


def test_fact_extraction_rejects_values_not_present_in_their_quote() -> None:
    facts, _ = _as_facts(
        {
            "facts": {
                "revenue": {
                    "value": "120",
                    "unit": "USD millions",
                    "period": "FY2024",
                    "source": "R1",
                    "quote": "Revenue was $100 million",
                }
            }
        },
        [
            {
                "source": "R1",
                "snippet": "Revenue was $120 million.",
            }
        ],
    )

    assert "revenue" not in facts


def test_fact_extraction_accepts_supported_values_across_table_whitespace() -> None:
    facts, _ = _as_facts(
        {
            "facts": {
                "revenue": {
                    "value": "32765",
                    "unit": "USD millions",
                    "period": "FY2018",
                    "source": "R1",
                    "quote": "Net sales $32,765",
                }
            }
        },
        [
            {
                "source": "R1",
                "snippet": "2018 Net sales $\n32,765.",
            }
        ],
    )

    assert facts["revenue"]["value"] == "32765"


def test_extractor_retries_malformed_json_once() -> None:
    class MalformedJsonLLM(FakeLLMClient):
        calls = 0

        def create_chat_completion(
            self,
            messages: list[dict[str, Any]],
            max_tokens: int = 256,
            model: str | None = None,
            temperature: float | None = None,
        ) -> dict[str, Any]:
            type(self).calls += 1
            content = (
                '{"facts":'
                if type(self).calls == 1
                else (
                    '{"facts":{"revenue":{"value":"120","unit":"USD millions",'
                    '"period":"FY2024","source":"R1",'
                    '"quote":"Revenue was $120 million"}}}'
                )
            )
            return {"choices": [{"message": {"content": content}}]}

    graph = FinanceReportGraph(
        rag_pipeline=FakeRAGPipeline(),
        llm_factory=MalformedJsonLLM,
    )

    result = graph._extract(
        {
            "query": "Summarize revenue growth",
            "evidence": [
                {
                    "source": "R1",
                    "document": "Example_2024_10K",
                    "page": 12,
                    "snippet": "Revenue was $120 million.",
                }
            ],
        }
    )

    assert MalformedJsonLLM.calls == 2
    assert result["extraction_facts"]["revenue"]["value"] == "120"


def test_report_extractor_fills_missing_facts_in_a_focused_pass() -> None:
    class SupplementalLLM(FakeLLMClient):
        calls = 0
        requests: list[list[dict[str, Any]]] = []

        def create_chat_completion(
            self,
            messages: list[dict[str, Any]],
            max_tokens: int = 256,
            model: str | None = None,
            temperature: float | None = None,
        ) -> dict[str, Any]:
            type(self).calls += 1
            type(self).requests.append(messages)
            content = (
                '{"facts":{"revenue":{"value":"120","unit":"USD millions",'
                '"period":"FY2024","source":"R1",'
                '"quote":"Revenue was $120 million"}}}'
                if type(self).calls == 1
                else (
                    '{"facts":{"operating_cash_flow":{"value":"30",'
                    '"unit":"USD millions","period":"FY2024","source":"R1",'
                    '"quote":"Net cash provided by operating activities $30"},'
                    '"capital_expenditures":{"value":"-10","unit":"USD millions",'
                    '"period":"FY2024","source":"R1",'
                    '"quote":"Purchases of property, plant and equipment (10)"}}}'
                )
            )
            return {"choices": [{"message": {"content": content}}]}

    graph = FinanceReportGraph(
        rag_pipeline=FakeRAGPipeline(),
        llm_factory=SupplementalLLM,
    )

    result = graph._extract(
        {
            "query": "Generate a FY2024 financial metrics report",
            "prompting_strategy": "few_shot",
            "fiscal_year": 2024,
            "report_mode": True,
            "evidence": [
                {
                    "source": "R1",
                    "document": "Example_2024_10K",
                    "page": 12,
                    "snippet": (
                        "Revenue was $120 million. Net cash provided by operating "
                        "activities $30. Purchases of property, plant and equipment (10)."
                    ),
                }
            ],
        }
    )

    assert SupplementalLLM.calls == 2
    assert all(
        "JSON shape" in request[0]["content"] and len(request) == 4
        for request in SupplementalLLM.requests
    )
    assert result["extraction_facts"]["operating_cash_flow"]["value"] == "30"
    assert result["extraction_facts"]["capital_expenditures"]["value"] == "-10"


def test_extractor_reports_rate_limit_without_failing_the_report() -> None:
    class RateLimitedLLM(FakeLLMClient):
        def create_chat_completion(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise rate_limit_error()

    graph = FinanceReportGraph(
        rag_pipeline=FakeRAGPipeline(),
        llm_factory=RateLimitedLLM,
    )
    result = graph._extract(
        {
            "query": "What was revenue?",
            "evidence": [
                {
                    "source": "R1",
                    "document": "Example_2024_10K",
                    "page": 12,
                    "snippet": "Revenue was $120 million.",
                }
            ],
        }
    )

    assert result["extraction_facts"] == {}
    assert result["data_status"]["fact_extraction"] == "rate_limited"


def test_supplemental_extraction_rate_limit_preserves_primary_facts() -> None:
    class SupplementalRateLimitedLLM(FakeLLMClient):
        calls = 0

        def create_chat_completion(
            self, *args: Any, **kwargs: Any
        ) -> dict[str, Any]:
            type(self).calls += 1
            if type(self).calls > 1:
                raise rate_limit_error()
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"facts":{"revenue":{"value":"120",'
                                '"unit":"USD millions","period":"FY2024",'
                                '"source":"R1","quote":"Revenue was $120 million"}}}'
                            )
                        }
                    }
                ]
            }

    graph = FinanceReportGraph(
        rag_pipeline=FakeRAGPipeline(),
        llm_factory=SupplementalRateLimitedLLM,
    )
    result = graph._extract(
        {
            "query": "Generate a FY2024 financial metrics report",
            "fiscal_year": 2024,
            "report_mode": True,
            "evidence": [
                {
                    "source": "R1",
                    "document": "Example_2024_10K",
                    "page": 12,
                    "snippet": "Revenue was $120 million.",
                }
            ],
        }
    )

    assert result["extraction_facts"]["revenue"]["value"] == "120"
    assert result["data_status"]["fact_extraction"] == "partial_rate_limited"


def test_synthesizer_retries_when_answer_omits_valid_citations() -> None:
    class CitationRetryLLM(FakeLLMClient):
        calls = 0

        def create_chat_completion(
            self,
            messages: list[dict[str, Any]],
            max_tokens: int = 256,
            model: str | None = None,
            temperature: float | None = None,
        ) -> dict[str, Any]:
            type(self).calls += 1
            answer = (
                "Revenue increased."
                if type(self).calls == 1
                else "Revenue increased based on the filing [R1]."
            )
            return {"choices": [{"message": {"content": answer}}]}

    graph = FinanceReportGraph(
        rag_pipeline=FakeRAGPipeline(),
        llm_factory=CitationRetryLLM,
    )

    result = graph._synthesize(
        {
            "query": "Summarize revenue growth",
            "evidence": [
                {
                    "source": "R1",
                    "document": "Example_2024_10K",
                    "page": 12,
                    "snippet": "Revenue was $120 million.",
                }
            ],
            "metrics": [],
        }
    )

    assert CitationRetryLLM.calls == 2
    assert "[R1]" in result["answer"]


def test_synthesizer_returns_cited_excerpts_when_model_is_rate_limited() -> None:
    class RateLimitedLLM(FakeLLMClient):
        def create_chat_completion(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise rate_limit_error()

    graph = FinanceReportGraph(
        rag_pipeline=FakeRAGPipeline(),
        llm_factory=RateLimitedLLM,
    )
    result = graph._synthesize(
        {
            "query": "Summarize revenue",
            "evidence": [
                {
                    "source": "R1",
                    "document": "Example_2024_10K",
                    "page": 12,
                    "snippet": "Revenue was $120 million.",
                }
            ],
            "metrics": [],
            "data_status": {},
        }
    )

    assert "[R1]" in result["answer"]
    assert "Revenue was $120 million." in result["answer"]
    assert (
        result["data_status"]["financial_analyst"]
        == "rate_limited_evidence_fallback"
    )


@pytest.mark.parametrize(
    ("strategy", "expected_prompt_text", "has_example"),
    [
        ("zero_shot", None, False),
        ("few_shot", "format demonstration only", True),
        ("cot", "step by step internally", False),
    ],
)
def test_synthesizer_uses_selected_prompting_strategy(
    strategy: str,
    expected_prompt_text: str | None,
    has_example: bool,
) -> None:
    class PromptCaptureLLM(FakeLLMClient):
        requests: list[list[dict[str, Any]]] = []

        def create_chat_completion(
            self,
            messages: list[dict[str, Any]],
            max_tokens: int = 256,
            model: str | None = None,
            temperature: float | None = None,
        ) -> dict[str, Any]:
            type(self).requests.append(messages)
            return {"choices": [{"message": {"content": "Revenue was $120 million [R1]."}}]}

    PromptCaptureLLM.requests = []
    graph = FinanceReportGraph(
        rag_pipeline=FakeRAGPipeline(),
        llm_factory=PromptCaptureLLM,
    )
    result = graph._synthesize(
        {
            "query": "Summarize revenue",
            "prompting_strategy": strategy,
            "evidence": [
                {
                    "source": "R1",
                    "document": "Example_2024_10K",
                    "page": 12,
                    "snippet": "Revenue was $120 million.",
                }
            ],
            "metrics": [],
        }
    )

    messages = PromptCaptureLLM.requests[0]
    assert result["answer"] == "Revenue was $120 million [R1]."
    assert (len(messages) == 4) is has_example
    if expected_prompt_text:
        assert expected_prompt_text in messages[0]["content"]


@pytest.mark.parametrize(
    ("strategy", "expected_prompt_text", "has_example"),
    [
        ("zero_shot", None, False),
        ("few_shot", "JSON shape", True),
        ("cot", "step by step internally", False),
    ],
)
def test_fact_extractor_uses_selected_prompting_strategy(
    strategy: str,
    expected_prompt_text: str | None,
    has_example: bool,
) -> None:
    class PromptCaptureLLM(FakeLLMClient):
        requests: list[list[dict[str, Any]]] = []

        def create_chat_completion(
            self,
            messages: list[dict[str, Any]],
            max_tokens: int = 256,
            model: str | None = None,
            temperature: float | None = None,
        ) -> dict[str, Any]:
            type(self).requests.append(messages)
            return super().create_chat_completion(
                messages, max_tokens=max_tokens, model=model, temperature=temperature
            )

    PromptCaptureLLM.requests = []
    graph = FinanceReportGraph(
        rag_pipeline=FakeRAGPipeline(),
        llm_factory=PromptCaptureLLM,
    )
    graph._extract(
        {
            "query": "What was revenue?",
            "prompting_strategy": strategy,
            "evidence": [
                {
                    "source": "R1",
                    "document": "Example_2024_10K",
                    "page": 12,
                    "snippet": "Revenue was $120 million.",
                }
            ],
        }
    )

    messages = PromptCaptureLLM.requests[0]
    assert (len(messages) == 4) is has_example
    if expected_prompt_text:
        assert expected_prompt_text in messages[0]["content"]
