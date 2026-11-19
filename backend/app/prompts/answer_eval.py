"""Test the answer-side prompts on FinanceBench questions with their gold evidence.

No database or retrieval is needed: each question comes with the exact evidence
passages FinanceBench annotated, given to the prompt as S1, S2, ... ("oracle
evidence"). This isolates prompt quality from retrieval quality.

Run from backend/ (uses .env via LLMClient), one node at a time:

    python -m app.prompts.answer_eval --node extract_financial_fact
    python -m app.prompts.answer_eval --node synthesize_answer --limit 30 --out synth_run1.json
    python -m app.prompts.answer_eval --node verify_answer
    python -m app.prompts.answer_eval --node repair_once
    python -m app.prompts.answer_eval --node analyze_financial_report

Data: data/evaluation/financebench_dev.jsonl (99 questions). The held-out split and
the project's 75-query golden set are refused unless --final-run is passed.

What is scored per node
-----------------------
extract   : valid JSON; gold number found among extracted facts (questions whose
            answer is a figure printed in the evidence).
analyze   : valid JSON; findings cite only supplied IDs; gold figure in findings.
synthesize: valid JSON; citations resolve; answered vs refused/insufficient;
            gold figure in the answer. For "numerical reasoning" questions no
            calculator results are supplied yet (the calculate_* tools are not
            built), so the ideal answer gives the cited inputs and status
            partially_answered; a correct computed figure there is reported
            separately as model arithmetic.
verify    : gold-based drafts, half correct and half with a planted wrong number;
            scores detection of the planted error and false alarms on correct drafts.
repair    : planted-error drafts plus a number_mismatch issue; scores whether the
            repaired answer has the gold figure and drops the planted one.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from . import PROMPT_SET_VERSION, Node, Technique, available, format_evidence, format_tool_results, get_prompt, render_citations
from .runner import run_prompt

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = ROOT / "data" / "evaluation" / "financebench_dev.jsonl"
EVIDENCE_CHAR_BUDGET = 9000  # keeps prompt + 1024 output tokens under the 8,192-token endpoint limit
NO_PLAN = '{"note": "plan not supplied in this test"}'

_NUM = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?")


# --------------------------------------------------------------------------- helpers

def load_items(path: Path, final_run: bool) -> List[Dict[str, Any]]:
    if "golden_75" in path.name:
        sys.exit("The 75-query golden set is not used for prompt testing.")
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if any(i.get("split") == "held_out" for i in items) and not final_run:
        sys.exit("This file contains held-out questions. Use financebench_dev.jsonl, or --final-run once prompts are frozen.")
    return items


def numbers_in(text: str) -> List[float]:
    out = []
    for m in _NUM.finditer(text or ""):
        tok = m.group(0)
        digits = re.sub(r"[^\d.]", "", tok)
        if not digits.strip(".") or digits.count(".") > 1:
            continue
        val = float(digits)
        out.append(-val if (tok.startswith("(") and tok.endswith(")")) or tok.startswith("-") else val)
    return out


def matches(gold: float, candidates: Iterable[float]) -> bool:
    """Same figure allowing rounding and a millions/billions/thousands change of scale."""
    for c in candidates:
        for scaled in (c, c * 1000, c / 1000, c * 1_000_000, c / 1_000_000):
            if abs(abs(scaled) - abs(gold)) <= max(0.006 * abs(gold), 0.011):
                return True
    return False


def gold_hit(item: Dict[str, Any], text: str) -> bool:
    golds = item.get("numeric_gold") or []
    text = re.sub(r"\[[SC]\d+\]", " ", text or "")  # citation markers are not figures
    return bool(golds) and all(matches(g, numbers_in(text)) for g in golds)


def gold_in_evidence(item: Dict[str, Any]) -> bool:
    return gold_hit(item, " ".join(e["text"] for e in item["evidence"]))


def is_numerical_reasoning(item: Dict[str, Any]) -> bool:
    return "numerical" in (item.get("question_reasoning") or "").lower()


def budget_evidence(evidence: Sequence[Dict[str, Any]], budget: int = EVIDENCE_CHAR_BUDGET) -> List[Dict[str, Any]]:
    total = sum(len(e["text"]) for e in evidence)
    if total <= budget:
        return [dict(e) for e in evidence]
    share = budget / total
    return [{**e, "text": e["text"][: max(400, int(len(e["text"]) * share))] + " [...]"} for e in evidence]


def plant_error(answer: str) -> Optional[str]:
    """Change the first figure in the answer by about +7% (keeps its format)."""
    m = re.search(r"\d[\d,]*(?:\.\d+)?", answer)
    if not m:
        return None
    raw = m.group(0)
    val = float(raw.replace(",", ""))
    if val == 0:
        return None
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    new = round(val * 1.07 + (1 if decimals == 0 and val < 15 else 0), decimals)
    new_txt = f"{new:,.{decimals}f}" if "," in raw else f"{new:.{decimals}f}"
    if new_txt == raw:
        return None
    return answer[: m.start()] + new_txt + answer[m.end():]


def draft_json(answer_text: str) -> str:
    return json.dumps({"status": "answered", "answer": f"{answer_text} [S1]",
                       "claims": [{"text": answer_text, "source_ids": ["S1"], "calc_ids": []}]})


def _mean(vals: Iterable[Any]) -> Optional[float]:
    v = [float(x) for x in vals if x is not None]
    return sum(v) / len(v) if v else None


# --------------------------------------------------------------------------- node tasks
# Each returns (variables for the prompt, scorer(parsed_data) -> metrics dict, extra context for the record)

def eligible(node: Node, item: Dict[str, Any]) -> bool:
    if node in (Node.VERIFY, Node.REPAIR, Node.EXTRACT):
        return bool(item.get("numeric_gold")) and gold_in_evidence(item)
    if node is Node.ANALYZE:
        return len(item["evidence"]) >= 2 or item["question_type"] != "metrics-generated"
    return True


def build_cases(node: Node, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    ev = budget_evidence(item["evidence"])
    evidence = format_evidence(ev)
    q = item["question"]
    if node is Node.EXTRACT:
        def score(d):
            facts = d.get("facts") or []
            values = " ".join(str(f.get("value", "")) for f in facts)
            return {"facts": len(facts), "gold_hit": gold_hit(item, values),
                    "cites_ok": all(f.get("source_id") in {e["source_id"] for e in ev} for f in facts)}
        return [{"case": "extract", "vars": {"metrics": f"whatever figures are needed to answer: {q}", "periods": "as stated in the question", "evidence": evidence}, "score": score}]

    if node is Node.ANALYZE:
        def score(d):
            findings = d.get("findings") or []
            ids = {s for f in findings for s in (f.get("source_ids") or [])}
            text = " ".join(str(f.get("statement", "")) for f in findings)
            return {"findings": len(findings), "cites_ok": bool(findings) and ids <= {e["source_id"] for e in ev},
                    "gold_hit": gold_hit(item, text) if item.get("numeric_gold") else None}
        return [{"case": "analyze", "vars": {"question": q, "plan": NO_PLAN, "evidence": evidence}, "score": score}]

    if node is Node.SYNTHESIZE:
        def score(d):
            rendered = render_citations(str(d.get("answer", "")), ev)
            status = d.get("status")
            answered = status in ("answered", "partially_answered", "corrected")
            cited = bool(rendered.cited_source_ids)
            hit = gold_hit(item, str(d.get("answer", ""))) if item.get("numeric_gold") else None
            needs_calc = is_numerical_reasoning(item) and bool(item.get("numeric_gold")) and not gold_in_evidence(item)
            return {"status": status, "answered": answered,
                    "cites_ok": not rendered.unknown_ids and (cited or not answered),
                    "gold_hit": None if needs_calc else hit,
                    "model_arithmetic": bool(hit) if needs_calc else None}
        return [{"case": "synthesize", "vars": {"question": q, "policy_note": "none", "alternative_question": "none",
                                                 "plan": NO_PLAN, "evidence": evidence, "tool_results": format_tool_results([])}, "score": score}]

    if node is Node.VERIFY:
        cases = []
        good = item["gold_answer"].strip()
        bad = plant_error(good)
        for label, text in (("correct", good), ("planted", bad)):
            if text is None:
                continue
            def score(d, label=label):
                verdict = d.get("verdict")
                types = {i.get("type") for i in (d.get("issues") or [])}
                return {"draft": label, "verdict": verdict,
                        "right": (verdict == "pass") if label == "correct" else (verdict == "fail"),
                        "number_issue": bool(types & {"number_mismatch", "unsupported_claim", "wrong_citation"}) if label == "planted" else None}
            cases.append({"case": f"verify_{label}", "vars": {"question": q, "evidence": evidence, "tool_results": format_tool_results([]), "draft": draft_json(text)}, "score": score})
        return cases

    if node is Node.REPAIR:
        good = item["gold_answer"].strip()
        bad = plant_error(good)
        if bad is None:
            return []
        planted_nums = [n for n in numbers_in(bad) if not matches(n, numbers_in(good))]
        issues = json.dumps([{"claim_index": 0, "type": "number_mismatch", "detail": "The figure in this claim does not match the cited evidence."}])
        def score(d):
            ans = str(d.get("answer", ""))
            return {"fixed": gold_hit(item, ans), "planted_removed": not any(matches(p, numbers_in(ans)) for p in planted_nums),
                    "cites_ok": not render_citations(ans, ev).unknown_ids}
        return [{"case": "repair", "vars": {"question": q, "evidence": evidence, "tool_results": format_tool_results([]),
                                             "draft": draft_json(bad), "issues": issues}, "score": score}]
    raise ValueError(f"{node.value} is not an answer-side node")


# --------------------------------------------------------------------------- run + report

def evaluate(llm: Any, node: Node, items: Sequence[Dict[str, Any]], techniques: Iterable[Technique],
             progress: Callable[[str], None] = lambda _m: None) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"prompt_set": PROMPT_SET_VERSION, "node": node.value, "techniques": {}, "rows": []}
    pool = [i for i in items if eligible(node, i)]
    summary["n_items"] = len(pool)
    for tech in techniques:
        spec = get_prompt(node, tech)
        recs = []
        for n, item in enumerate(pool, start=1):
            for case in build_cases(node, item):
                try:
                    res = run_prompt(llm, spec, **case["vars"])
                    metrics = case["score"](res.parsed.data) if res.ok else {}
                    rec = {"question_id": item["question_id"], "case": case["case"], "prompt_id": spec.prompt_id,
                           "valid": res.ok, "json_first_try": res.ok and res.calls == 1, "calls": res.calls,
                           "latency_s": round(res.latency_s, 3), "retries": res.retries,
                           "tokens": sum(int(u.get("total_tokens", 0) or 0) for u in res.usage),
                           **metrics, "output": res.parsed.data, "error": res.parsed.error}
                except Exception as exc:  # keep going; one bad call must not lose the run
                    rec = {"question_id": item["question_id"], "case": case["case"], "prompt_id": spec.prompt_id,
                           "valid": False, "json_first_try": False, "calls": 0, "latency_s": None, "tokens": 0,
                           "output": None, "error": f"{type(exc).__name__}: {exc}"}
                recs.append(rec)
                summary["rows"].append(rec)
            progress(f"{tech.value}: {n}/{len(pool)}")
        summary["techniques"][tech.value] = _summarize(node, recs)
    return summary


def _summarize(node: Node, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in recs if r["valid"]]
    s: Dict[str, Any] = {
        "cases": len(recs),
        "valid": _mean(r["valid"] for r in recs),
        "json_first_try": _mean(r["json_first_try"] for r in recs),
        "p50_latency_s": statistics.median([r["latency_s"] for r in recs if r["latency_s"] is not None] or [0]),
        "avg_tokens": _mean(r["tokens"] for r in recs),
    }
    if node is Node.EXTRACT:
        s.update(gold_hit=_mean(r.get("gold_hit") for r in ok), cites_ok=_mean(r.get("cites_ok") for r in ok))
    elif node is Node.ANALYZE:
        s.update(cites_ok=_mean(r.get("cites_ok") for r in ok), gold_hit=_mean(r.get("gold_hit") for r in ok),
                 has_findings=_mean((r.get("findings") or 0) > 0 for r in ok))
    elif node is Node.SYNTHESIZE:
        s.update(answered=_mean(r.get("answered") for r in ok), cites_ok=_mean(r.get("cites_ok") for r in ok),
                 gold_hit=_mean(r.get("gold_hit") for r in ok), model_arithmetic=_mean(r.get("model_arithmetic") for r in ok))
    elif node is Node.VERIFY:
        s.update(accuracy=_mean(r.get("right") for r in ok),
                 catches_planted=_mean(r.get("right") for r in ok if r["case"] == "verify_planted"),
                 passes_correct=_mean(r.get("right") for r in ok if r["case"] == "verify_correct"))
    elif node is Node.REPAIR:
        s.update(fixed=_mean(r.get("fixed") for r in ok), planted_removed=_mean(r.get("planted_removed") for r in ok),
                 cites_ok=_mean(r.get("cites_ok") for r in ok))
    return s


_LABELS = {
    "valid": "valid", "json_first_try": "JSON 1st try", "gold_hit": "gold figure", "cites_ok": "citations ok",
    "answered": "answered", "model_arithmetic": "model arithmetic", "has_findings": "has findings",
    "accuracy": "accuracy", "catches_planted": "catches planted error", "passes_correct": "passes correct draft",
    "fixed": "fixed", "planted_removed": "planted removed", "p50_latency_s": "p50 latency s", "avg_tokens": "avg tokens",
}


def to_markdown(summary: Dict[str, Any]) -> str:
    techs = summary["techniques"]
    if not techs:
        return "No results."
    keys = [k for k in next(iter(techs.values())) if k != "cases"]
    head = "| technique | cases | " + " | ".join(_LABELS.get(k, k) for k in keys) + " |"
    sep = "|" + "---|" * (len(keys) + 2)
    lines = [f"Node `{summary['node']}`: {summary['n_items']} FinanceBench dev questions, prompt set {summary['prompt_set']}", "", head, sep]
    for tech, m in techs.items():
        cells = []
        for k in keys:
            v = m[k]
            if v is None:
                cells.append("-")
            elif k == "p50_latency_s":
                cells.append(f"{v:.2f}")
            elif k == "avg_tokens":
                cells.append(f"{v:.0f}")
            else:
                cells.append(f"{v * 100:.1f}%")
        lines.append(f"| {tech} | {m['cases']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


ANSWER_NODES = [Node.EXTRACT, Node.ANALYZE, Node.SYNTHESIZE, Node.VERIFY, Node.REPAIR]


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node", required=True, choices=[n.value for n in ANSWER_NODES])
    p.add_argument("--techniques", nargs="*", choices=[t.value for t in Technique])
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--limit", type=int, default=25, help="questions per technique (default 25; 0 = all eligible)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--final-run", action="store_true", help="allow held-out data (once, after prompts are frozen)")
    args = p.parse_args(argv)

    node = Node(args.node)
    techs = [Technique(t) for t in args.techniques] if args.techniques else available(node)
    items = load_items(args.data, args.final_run)
    pool = [i for i in items if eligible(node, i)]
    if args.limit:
        pool = pool[: args.limit]
    print(f"{node.value}: {len(pool)} questions x {len(techs)} technique(s). This calls the model one request at a time.", flush=True)

    from app.services.llm_client import LLMClient  # lazy: tests run without config

    with LLMClient() as llm:
        summary = evaluate(llm, node, pool, techs, progress=lambda m: print("  " + m, end="\r", flush=True))
    print(" " * 40)
    print(to_markdown(summary))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\nFull results: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
