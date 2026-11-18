"""Offline tests for the prompt library (no network, fake LLM)."""
from __future__ import annotations

import json

import pytest

from app.prompts import (
    DEFAULT_TECHNIQUE,
    Node,
    Technique,
    all_prompts,
    available,
    format_evidence,
    format_tool_results,
    get_prompt,
    parse_output,
    refusal_message,
    render_citations,
    run_prompt,
)
from app.prompts.experiment import DEFAULT_DEV, evaluate, load_jsonl, to_markdown

DUMMY = {
    "question": "What was revenue?",
    "company": "none",
    "period": "none",
    "plan": "{}",
    "metrics": "revenue",
    "periods": "FY2024",
    "evidence": format_evidence([{"source_id": "S1", "text": "Revenue was $10M."}]),
    "tool_results": format_tool_results([]),
    "draft": "{}",
    "issues": "[]",
    "policy_note": "none",
    "alternative_question": "none",
}


class FakeLLM:
    model = "Qwen/Qwen3-8B"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create_chat_completion(self, messages, max_tokens=256, temperature=None):
        self.calls.append(messages)
        text = self.replies.pop(0) if self.replies else "{}"
        return {"choices": [{"message": {"content": text}}], "usage": {"total_tokens": 10}}


@pytest.mark.parametrize("spec", all_prompts(), ids=lambda s: s.prompt_id)
def test_every_prompt_renders(spec):
    msgs = spec.render(model_name="qwen2.5:3b", **{k: DUMMY[k] for k in spec.variables})
    assert msgs[0]["role"] == "system" and msgs[-1]["role"] == "user"
    assert len(msgs) == 2 + 2 * len(spec.examples)
    assert "{{" not in msgs[-1]["content"]


@pytest.mark.parametrize("spec", all_prompts(), ids=lambda s: s.prompt_id)
def test_few_shot_examples_are_valid_and_leak_free(spec):
    for ex in spec.examples:
        parsed = parse_output(ex.assistant, spec)
        assert parsed.ok, (spec.prompt_id, parsed.error)
        # Golden-set company must never appear in prompts (avoid eval leakage).
        assert "acme" not in (ex.user + ex.assistant).lower()
    assert "acme" not in (spec.system + spec.user_template).lower()


def test_every_node_has_its_default_technique():
    for node, tech in DEFAULT_TECHNIQUE.items():
        assert get_prompt(node).technique is tech
    assert set(available(Node.CLASSIFY)) == {Technique.ZERO_SHOT, Technique.FEW_SHOT, Technique.COT}


def test_missing_variable_fails_closed():
    with pytest.raises(KeyError):
        get_prompt(Node.CLASSIFY).render(question="x")


def test_qwen3_soft_switch(monkeypatch):
    spec = get_prompt(Node.GUARDRAIL)
    vars_ = {k: DUMMY[k] for k in spec.variables}
    assert spec.render(model_name="Qwen/Qwen3-8B", **vars_)[-1]["content"].endswith("/no_think")
    assert not spec.render(model_name="qwen2.5:3b", **vars_)[-1]["content"].endswith("think")
    monkeypatch.setenv("PROMPT_QWEN3_THINKING", "1")
    assert spec.render(model_name="Qwen/Qwen3-8B", **vars_)[-1]["content"].endswith("/think")


def test_env_override_technique(monkeypatch):
    monkeypatch.setenv("PROMPT_TECHNIQUE_CLASSIFY", "cot")
    assert get_prompt(Node.CLASSIFY).technique is Technique.COT


def test_parse_handles_think_fences_and_prose():
    spec = get_prompt(Node.GUARDRAIL, "zero_shot")
    raw = '<think>hmm {"x":1}</think>Sure!\n```json\n{"decision": "allow", "category": "none", "reason": "ok {fine}"}\n```'
    res = parse_output(raw, spec)
    assert res.ok and res.data["reason"] == "ok {fine}"


def test_parse_rejects_bad_enum_and_missing_keys():
    spec = get_prompt(Node.GUARDRAIL, "zero_shot")
    assert "must be one of" in parse_output('{"decision": "maybe", "category": "none", "reason": "x"}', spec).error
    assert "Missing" in parse_output('{"decision": "allow"}', spec).error
    assert parse_output("no json here", spec).error


def test_run_prompt_does_one_format_repair():
    spec = get_prompt(Node.GUARDRAIL)
    llm = FakeLLM(["oops", json.dumps({"decision": "refuse", "category": "investment_advice", "reason": "x"})])
    res = run_prompt(llm, spec, question="Buy?", company="none", period="none")
    assert res.ok and res.calls == 2 and len(llm.calls) == 2
    assert "could not be used" in llm.calls[1][-1]["content"]


def test_evidence_formatting_escapes_and_handles_empty():
    out = format_evidence([{"source_id": "S1", "document_name": 'X "10-K"', "page": 3, "text": "a"}])
    assert 'source_id="S1"' in out and "X '10-K'" in out and 'page="3"' in out
    assert "NO EVIDENCE" in format_evidence([])


def test_refusal_messages():
    msg = refusal_message("investment_advice")
    assert msg.startswith("REFUSAL: ") and "buy, sell, or hold" in msg
    assert "outside" in refusal_message("unknown_cat")
    # every guardrail refusal category has fixed text
    from app.prompts.routing import GUARD_CATEGORIES
    from app.prompts.refusals import REFUSAL_TEXT
    assert set(GUARD_CATEGORIES) - {"none"} <= set(REFUSAL_TEXT)


def test_render_citations_spec_format_and_unknown_ids():
    evidence = [
        {"source_id": "S1", "document_name": "Q2 FY25 10-Q", "section": "Statements of Operations", "page": 4, "period": "Q2 FY2025", "text": "x"},
        {"source_id": "S2", "document_name": "FY24 10-K", "page": 54, "text": "y"},
    ]
    calcs = [{"calc_id": "C1", "source_ids": ["S1"]}]
    out = render_citations("Sales grew 11.65% [S1][C1]. Other [S2]. Bad [S9][C7].", evidence, calcs)
    assert "[Q2 FY25 10-Q | Statements of Operations, p. 4 | Q2 FY2025]" in out.text
    assert out.text.count("Q2 FY25 10-Q") == 1  # S1 and C1 collapse to one citation
    assert "[FY24 10-K | p. 54]" in out.text
    assert out.unknown_ids == ["C7", "S9"] and out.cited_source_ids == ["S1", "S2"]


def test_synthesis_examples_cover_spec_answer_types():
    spec = get_prompt(Node.SYNTHESIZE, "few_shot")
    statuses = {json.loads(ex.assistant)["status"] for ex in spec.examples}
    assert {"answered", "corrected", "insufficient_evidence", "refused"} <= statuses
    refused = [json.loads(ex.assistant) for ex in spec.examples if json.loads(ex.assistant)["status"] == "refused"]
    assert all(r["answer"].startswith("REFUSAL: ") for r in refused)


def test_golden_set_is_held_out_and_complete():
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "data" / "evaluation" / "golden_75.jsonl"
    if not path.exists():
        pytest.skip("golden set not present")
    rows = load_jsonl(path)
    assert [r["question_id"] for r in rows] == [f"Q{i:03d}" for i in range(1, 76)]
    assert all(r["split"] == "held_out" for r in rows)
    assert {r["category"] for r in rows} == {"extraction", "calculation", "multi_hop", "refusal", "edge_case"}
    # no golden question text appears in any prompt
    blob = " ".join(s.system + s.user_template + " ".join(e.user + e.assistant for e in s.examples) for s in all_prompts()).lower()
    assert not any(r["question"].lower()[:60] in blob for r in rows)


def test_experiment_refuses_held_out(tmp_path):
    from app.prompts.experiment import main
    f = tmp_path / "h.jsonl"
    f.write_text(json.dumps({"question_id": "Q1", "question": "x", "expected_decision": "allow", "split": "held_out"}) + "\n")
    with pytest.raises(SystemExit):
        main(["--node", "pre_guardrail", "--dev", str(f)])


def test_experiment_scoring_with_fake_llm():
    items = [i for i in load_jsonl(DEFAULT_DEV) if i.get("expected_decision")][:2]
    replies = [json.dumps({"decision": i["expected_decision"], "category": i["expected_category"], "reason": "r"}) for i in items]
    summary = evaluate(FakeLLM(replies), Node.GUARDRAIL, items, [Technique.ZERO_SHOT])
    m = summary["techniques"]["zero_shot"]
    assert m["accuracy"] == 1.0 and m["json_first_try"] == 1.0
    assert "zero_shot" in to_markdown(summary)
