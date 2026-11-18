"""A/B prompting techniques on a labelled dev set.

Scores the two nodes that can be evaluated without the RAG pipeline:
  * pre_guardrail      -> expected_decision (allow/refuse/clarify), expected_category
  * classify_and_plan  -> expected_intent

Run from backend/ (uses .env via LLMClient):

    python -m app.prompts.experiment --node pre_guardrail --dev app/prompts/dev_sample.jsonl
    python -m app.prompts.experiment --node classify_and_plan --techniques zero_shot few_shot cot --repeats 3

Use ONLY the dev split for tuning. Freeze prompts before the held-out 75-query run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from . import PROMPT_SET_VERSION, Node, Technique, available, get_prompt
from .runner import run_prompt

DEFAULT_DEV = Path(__file__).with_name("dev_sample.jsonl")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score_item(node: Node, item: Dict[str, Any], data: Optional[Dict[str, Any]]) -> Dict[str, Optional[bool]]:
    if data is None:
        return {"correct": False, "category_correct": None}
    if node is Node.GUARDRAIL:
        correct = data.get("decision") == item.get("expected_decision")
        cat = None
        if item.get("expected_category"):
            cat = data.get("category") == item["expected_category"]
        return {"correct": correct, "category_correct": cat}
    if node is Node.CLASSIFY:
        return {"correct": data.get("intent") == item.get("expected_intent"), "category_correct": None}
    raise ValueError(f"No scorer for {node.value}")


def evaluate(
    llm: Any,
    node: Node,
    items: Sequence[Dict[str, Any]],
    techniques: Iterable[Technique],
    repeats: int = 1,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"prompt_set": PROMPT_SET_VERSION, "node": node.value, "n_items": len(items), "techniques": {}}
    rows: List[Dict[str, Any]] = []
    for tech in techniques:
        spec = get_prompt(node, tech)
        recs = []
        for item in items:
            for r in range(repeats):
                res = run_prompt(
                    llm,
                    spec,
                    question=item["question"],
                    company=item.get("company") or "none",
                    period=item.get("period") or "none",
                )
                s = score_item(node, item, res.parsed.data if res.ok else None)
                tokens = sum(int(u.get("total_tokens", 0) or 0) for u in res.usage)
                rec = {"question_id": item.get("question_id"), "repeat": r, "prompt_id": spec.prompt_id,
                       "json_ok_first_try": res.calls == 1 and res.ok, "valid": res.ok, "calls": res.calls,
                       "latency_s": round(res.latency_s, 3), "tokens": tokens, **s,
                       "output": res.parsed.data, "error": res.parsed.error}
                recs.append(rec)
                rows.append(rec)
        cat_vals = [x["category_correct"] for x in recs if x["category_correct"] is not None]
        summary["techniques"][tech.value] = {
            "prompt_id": spec.prompt_id,
            "accuracy": _mean(x["correct"] for x in recs),
            "category_accuracy": _mean(cat_vals) if cat_vals else None,
            "json_first_try": _mean(x["json_ok_first_try"] for x in recs),
            "valid_after_repair": _mean(x["valid"] for x in recs),
            "avg_calls": _mean(x["calls"] for x in recs),
            "p50_latency_s": statistics.median(x["latency_s"] for x in recs) if recs else None,
            "avg_tokens": _mean(x["tokens"] for x in recs),
        }
    summary["rows"] = rows
    return summary


def to_markdown(summary: Dict[str, Any]) -> str:
    head = "| technique | accuracy | category acc | JSON 1st try | valid | avg calls | p50 latency s | avg tokens |\n|---|---|---|---|---|---|---|---|"
    lines = [f"Node `{summary['node']}` — {summary['n_items']} items — prompt set {summary['prompt_set']}", "", head]
    for tech, m in summary["techniques"].items():
        lines.append(
            f"| {tech} | {_pct(m['accuracy'])} | {_pct(m['category_accuracy'])} | {_pct(m['json_first_try'])} | "
            f"{_pct(m['valid_after_repair'])} | {_num(m['avg_calls'])} | {_num(m['p50_latency_s'])} | {_num(m['avg_tokens'], 0)} |"
        )
    return "\n".join(lines)


def _mean(vals: Iterable[Any]) -> Optional[float]:
    v = [float(x) for x in vals]
    return sum(v) / len(v) if v else None


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def _num(x: Optional[float], d: int = 2) -> str:
    return "-" if x is None else f"{x:.{d}f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node", required=True, choices=[Node.GUARDRAIL.value, Node.CLASSIFY.value])
    p.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    p.add_argument("--techniques", nargs="*", choices=[t.value for t in Technique])
    p.add_argument("--repeats", type=int, default=1, help="repeat each item to measure stability")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="write full JSON results here")
    p.add_argument("--final-run", action="store_true",
                   help="allow held-out records (the 75-query golden set). Use ONCE, after prompts are frozen.")
    args = p.parse_args(argv)

    node = Node(args.node)
    techs = [Technique(t) for t in args.techniques] if args.techniques else available(node)
    items = load_jsonl(args.dev)
    if any(i.get("split") == "held_out" for i in items) and not args.final_run:
        sys.exit(
            "This file contains held-out (golden) questions. Tuning prompts on them would inflate the final score.\n"
            "Use a dev file, or pass --final-run once prompts are frozen."
        )
    items = [i for i in items if (i.get("expected_decision") if node is Node.GUARDRAIL else i.get("expected_intent"))]
    if args.limit:
        items = items[: args.limit]

    from app.services.llm_client import LLMClient  # imported lazily so tests need no config

    with LLMClient() as llm:
        summary = evaluate(llm, node, items, techs, args.repeats)
    print(to_markdown(summary))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\nFull results: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
