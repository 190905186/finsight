"""Offline test of the end-to-end runner: fake retriever + fake model (no DB, no network)."""
from __future__ import annotations

import json

import pytest

from app.prompts import answer_eval as ae
from app.prompts import e2e_eval as e2e

pytestmark = pytest.mark.skipif(not ae.DEFAULT_DATA.exists(), reason="FinanceBench dev split not present")


class FakeRetriever:
    def __init__(self, item, hit=True):
        self.item, self.hit = item, hit

    def retrieve(self, query, top_k=5, filters=None):
        assert filters == {"doc_name": self.item["doc_name"]}
        ev = self.item["evidence"][0]
        page = ev["page"] if self.hit else 9999
        return [{"chunk_id": f"c{len(query)}", "document": self.item["doc_name"], "page": page, "snippet": ev["text"][:800]}]


class ScriptedLLM:
    """Answers each node by recognising its system prompt."""
    model = "Qwen/Qwen3-8B"

    def __init__(self, gold):
        self.gold, self.calls = gold, 0

    def create_chat_completion(self, messages, max_tokens=256, temperature=None):
        self.calls += 1
        sys_text = messages[0]["content"]
        if "decide whether a user question may proceed" in sys_text:
            out = {"decision": "allow", "category": "none", "reason": "r", "alternative_question": None}
        elif "classify an in-scope" in sys_text:
            out = {"intent": "extraction", "company": "X", "periods": [], "metrics": [], "basis": "GAAP", "doc_types": [], "tools": ["search_filings"], "clarification_question": None}
        elif "rewrite a user question" in sys_text:
            out = {"queries": ["q one", "query two"]}
        elif "write the final answer" in sys_text:
            out = {"reasoning": ["x"], "status": "answered", "answer": f"{self.gold} [S1][S9]", "claims": [{"text": "t", "source_ids": ["S1"], "calc_ids": []}]}
        elif "audit a draft answer" in sys_text:
            out = {"reasoning": ["x"], "verdict": "pass", "issues": []}
        elif "fix a draft answer" in sys_text:
            out = {"status": "answered", "answer": f"{self.gold} [S1]", "claims": [{"text": "t", "source_ids": ["S1"], "calc_ids": []}]}
        else:  # judge
            out = {"verdict": "correct", "reason": "r"}
        return {"choices": [{"message": {"content": json.dumps(out)}}], "usage": {"total_tokens": 1}}


def test_chain_repairs_unknown_citation_and_scores():
    item = next(i for i in ae.load_items(ae.DEFAULT_DATA, False) if i["numeric_gold"] and ae.gold_in_evidence(i))
    llm = ScriptedLLM(item["gold_answer"])
    rec = e2e.run_question(llm, FakeRetriever(item), item)
    assert rec["guardrail"] == "allow" and rec["queries"] == ["q one", "query two"]
    assert rec["hit_raw"] and rec["hit_rewrite"] and rec["hit_merged"]
    assert rec["unknown_ids"] == ["S9"] and rec["repaired"] and rec["citations_ok"]  # invented S9 -> repair
    assert rec["gold_figure"] and rec["judge"] == "correct"
    assert rec["calls"] == 6 and llm.calls == 7  # 6 chain calls + judge
    s = e2e.summarize([rec])
    assert s["page_hit_merged"] == 1.0 and s["citations_ok"] == 1.0
    assert "page hit merged" in e2e.to_markdown({"prompt_set": "x", **s})


def test_page_miss_is_reported():
    item = ae.load_items(ae.DEFAULT_DATA, False)[0]
    rec = e2e.run_question(ScriptedLLM(item["gold_answer"]), FakeRetriever(item, hit=False), item, use_judge=False)
    assert rec["hit_raw"] is False and rec["hit_merged"] is False and "judge" not in rec
