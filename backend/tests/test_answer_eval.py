"""Offline tests for the answer-side harness (fake LLM, no network)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.prompts import Node, Technique, all_prompts
from app.prompts import answer_eval as ae

DEV = ae.DEFAULT_DATA
pytestmark = pytest.mark.skipif(not DEV.exists(), reason="FinanceBench dev split not present")


class FakeLLM:
    model = "Qwen/Qwen3-8B"

    def __init__(self, reply):
        self.reply = reply  # callable(messages) -> str

    def create_chat_completion(self, messages, max_tokens=256, temperature=None):
        return {"choices": [{"message": {"content": self.reply(messages)}}], "usage": {"total_tokens": 5}}


def _items():
    return ae.load_items(DEV, final_run=False)


def test_split_files_are_separate_and_locked():
    dev = _items()
    held = [json.loads(l) for l in open(DEV.with_name("financebench_holdout.jsonl"), encoding="utf-8")]
    assert len(dev) + len(held) == 150 and not ({d["question_id"] for d in dev} & {h["question_id"] for h in held})
    assert all(h["split"] == "held_out" for h in held)
    assert "financebench_id_03029" in {h["question_id"] for h in held}  # used by RAG tests
    with pytest.raises(SystemExit):
        ae.load_items(DEV.with_name("financebench_holdout.jsonl"), final_run=False)
    with pytest.raises(SystemExit):
        ae.load_items(DEV.with_name("golden_75.jsonl"), final_run=True)


def test_number_matching_and_planting():
    assert ae.matches(1577, ae.numbers_in("Capex was $1,577 million"))
    assert ae.matches(1577, ae.numbers_in("about $1.577 billion"))
    assert not ae.matches(1577, ae.numbers_in("$1,688 million"))
    assert ae.plant_error("$1577.00") == "$1687.39"
    assert ae.plant_error("no numbers") is None


def test_prompts_fit_8192_token_budget_with_largest_evidence():
    items = _items()
    biggest = max(items, key=lambda i: sum(len(e["text"]) for e in i["evidence"]))
    for node in ae.ANSWER_NODES:
        for case in ae.build_cases(node, biggest):
            for spec in [s for s in all_prompts() if s.node is node]:
                chars = sum(len(m["content"]) for m in spec.render(model_name="qwen3", **case["vars"]))
                assert chars / 3.2 + spec.max_tokens < 8192, (spec.prompt_id, chars)


def test_synthesize_scoring_with_fake_llm():
    items = [i for i in _items() if i["numeric_gold"] and ae.gold_in_evidence(i)][:3]
    def reply(messages):
        return json.dumps({"status": "answered", "answer": "The figure is 987654.321 [S1].", "claims": []})
    s = ae.evaluate(FakeLLM(reply), Node.SYNTHESIZE, items, [Technique.ZERO_SHOT])
    m = s["techniques"]["zero_shot"]
    assert m["valid"] == 1.0 and m["cites_ok"] == 1.0 and m["gold_hit"] == 0.0
    assert "zero_shot" in ae.to_markdown(s)


def test_verify_scoring_distinguishes_correct_and_planted():
    items = [i for i in _items() if ae.eligible(Node.VERIFY, i)][:4]
    def reply(messages):  # a perfect verifier: fail exactly the planted drafts
        draft = messages[-1]["content"].split("Draft answer JSON:")[1]
        planted = any(ae.plant_error(i["gold_answer"].strip()) and ae.plant_error(i["gold_answer"].strip()) in draft for i in items)
        return json.dumps({"verdict": "fail", "issues": [{"claim_index": 0, "type": "number_mismatch", "detail": "x"}]}) if planted else json.dumps({"verdict": "pass", "issues": []})
    m = ae.evaluate(FakeLLM(reply), Node.VERIFY, items, [Technique.ZERO_SHOT])["techniques"]["zero_shot"]
    assert m["catches_planted"] == 1.0 and m["passes_correct"] == 1.0


def test_harness_survives_endpoint_errors():
    items = _items()[:2]
    def reply(messages):
        raise RuntimeError("boom")
    s = ae.evaluate(FakeLLM(reply), Node.SYNTHESIZE, items, [Technique.ZERO_SHOT])
    assert s["techniques"]["zero_shot"]["valid"] == 0.0 and all("boom" in r["error"] for r in s["rows"])


def test_balanced_selection_and_reference_calc():
    items = _items()
    pick = ae.select(items, 9)
    assert {i["question_type"] for i in pick} == {"domain-relevant", "metrics-generated", "novel-generated"}
    for i in items:
        calc = ae.reference_calc(i)
        assert bool(calc) == ae.needs_calc(i)
        if calc:
            assert calc[0]["calc_id"] == "C1" and calc[0]["result"] == i["gold_answer"]
    assert not ae.matches(0.83, [830.0])  # small ratios never rescale


def test_synthesis_prompt_forbids_invented_calc_markers():
    from app.prompts import get_prompt
    for t in ("zero_shot", "few_shot", "cot"):
        assert "ONLY for a calc_id that appears" in get_prompt(Node.SYNTHESIZE, t).system


def test_judge_parses_verdicts():
    llm = FakeLLM(lambda m: '<think>x</think>{"verdict": "partially_correct", "reason": "r"}')
    assert ae.judge(llm, "q", "ref", "cand") == "partially_correct"
    assert ae.judge(FakeLLM(lambda m: "nonsense"), "q", "ref", "cand") is None
