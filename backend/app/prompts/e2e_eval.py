"""End-to-end test of the prompt chain with real retrieval (needs the RAG database).

For each FinanceBench dev question (balanced sample, never held-out or golden data):

    pre_guardrail -> classify_and_plan -> query_rewrite -> HybridRetriever (pgvector + BM25)
    -> synthesize_answer -> render_citations -> verify_answer -> repair_once (at most once)
    -> AI judge against FinanceBench's reference answer

Each node uses its default technique (DEFAULT_TECHNIQUE in base.py). Retrieval is filtered to
the question's document, as if the user had selected it in the UI. It also measures whether
query_rewrite finds the gold evidence page better than searching with the raw question.

Run INSIDE the backend container (the Windows host cannot load the Postgres driver):

    docker compose run --rm backend python -m app.prompts.e2e_eval --limit 24 --out /data/evaluation/runs/e2e_run1.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import PROMPT_SET_VERSION, Node, format_evidence, format_tool_results, get_prompt, render_citations, refusal_message
from .answer_eval import DEFAULT_DATA, budget_evidence, gold_hit, judge, load_items, reference_calc, select
from .runner import run_prompt

TOP_K_PER_QUERY = 5
MAX_EVIDENCE = 8  # e2e run 1: 6 dropped pages the rewritten queries had found


def _gold_pages(item: Dict[str, Any]) -> set:
    return {e["page"] for e in item["evidence"] if e.get("page")}


def _page_hit(item: Dict[str, Any], chunks: Sequence[Dict[str, Any]]) -> bool:
    gold = _gold_pages(item)
    return any(c.get("document") == item["doc_name"] and c.get("page") in gold for c in chunks)


def _merge(results: Sequence[Sequence[Dict[str, Any]]], limit: int) -> List[Dict[str, Any]]:
    """Round-robin merge of several ranked lists, de-duplicated by chunk id."""
    seen, out = set(), []
    for rank in range(max((len(r) for r in results), default=0)):
        for r in results:
            if rank < len(r) and r[rank]["chunk_id"] not in seen and len(out) < limit:
                seen.add(r[rank]["chunk_id"])
                out.append(r[rank])
    return out


def run_question(llm: Any, retriever: Any, item: Dict[str, Any], use_judge: bool = True) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rec: Dict[str, Any] = {"question_id": item["question_id"], "question_type": item["question_type"], "calls": 0}
    q, company = item["question"], item["company"]

    g = run_prompt(llm, get_prompt(Node.GUARDRAIL), question=q, company=company, period="none")
    rec["calls"] += g.calls
    rec["guardrail"] = (g.parsed.data or {}).get("decision") if g.ok else "invalid"
    if rec["guardrail"] == "refuse":  # every FinanceBench dev question should be allowed
        rec.update(final_status="refused", answer=refusal_message((g.parsed.data or {}).get("category", "out_of_scope")))

    c = run_prompt(llm, get_prompt(Node.CLASSIFY), question=q, company=company, period="none")
    rec["calls"] += c.calls
    plan = c.parsed.data if c.ok else {}
    rec["intent"] = plan.get("intent")

    w = run_prompt(llm, get_prompt(Node.REWRITE), question=q, plan=json.dumps(plan))
    rec["calls"] += w.calls
    queries = [x for x in ((w.parsed.data or {}).get("queries") or []) if isinstance(x, str) and x.strip()][:3] if w.ok else []
    rec["queries"] = queries

    filters = {"doc_name": item["doc_name"]}
    raw = retriever.retrieve(q, top_k=TOP_K_PER_QUERY, filters=filters)
    rewritten = [retriever.retrieve(x, top_k=TOP_K_PER_QUERY, filters=filters) for x in queries]
    chunks = _merge([*rewritten, raw], MAX_EVIDENCE)  # rewritten queries first: they hit the gold page far more often
    rec["hit_raw"] = _page_hit(item, raw)
    rec["hit_rewrite"] = _page_hit(item, [x for r in rewritten for x in r]) if rewritten else None
    rec["hit_merged"] = _page_hit(item, chunks)

    evidence = budget_evidence([
        {"source_id": f"S{i}", "document_name": ch["document"], "page": ch["page"],
         "period": next((e["period"] for e in item["evidence"] if e.get("period")), ""), "text": ch["snippet"]}
        for i, ch in enumerate(chunks, start=1)
    ])
    calcs = reference_calc(item)  # stand-in for the calculate_* tools (not built yet)
    ev_text, tools_text = format_evidence(evidence), format_tool_results(calcs)

    if rec["guardrail"] != "refuse":
        s = run_prompt(llm, get_prompt(Node.SYNTHESIZE), question=q, policy_note="none", alternative_question="none",
                       plan=json.dumps(plan), evidence=ev_text, tool_results=tools_text)
        rec["calls"] += s.calls
        draft = s.parsed.data if s.ok else None
        repaired = False
        if draft:
            rendered = render_citations(str(draft.get("answer", "")), evidence, calcs)
            v = run_prompt(llm, get_prompt(Node.VERIFY), question=q, evidence=ev_text, tool_results=tools_text, draft=json.dumps(draft))
            rec["calls"] += v.calls
            verdict = (v.parsed.data or {}).get("verdict") if v.ok else "invalid"
            rec["verify"] = verdict
            rec["unknown_ids"] = rendered.unknown_ids
            if verdict == "fail" or rendered.unknown_ids:
                issues = list((v.parsed.data or {}).get("issues") or [])
                if rendered.unknown_ids:
                    issues.append({"claim_index": 0, "type": "wrong_citation", "detail": f"IDs not supplied in this message: {rendered.unknown_ids}"})
                r = run_prompt(llm, get_prompt(Node.REPAIR), question=q, evidence=ev_text, tool_results=tools_text,
                               draft=json.dumps(draft), issues=json.dumps(issues))
                rec["calls"] += r.calls
                if r.ok:
                    draft, repaired = r.parsed.data, True
            final = render_citations(str(draft.get("answer", "")), evidence, calcs)
            rec.update(final_status=draft.get("status"), answer=final.text, repaired=repaired,
                       citations_ok=not final.unknown_ids)
        else:
            rec.update(final_status="invalid", answer="", repaired=False, citations_ok=False)

    if item.get("numeric_gold"):
        rec["gold_figure"] = gold_hit(item, rec.get("answer", ""))
    if use_judge:
        rec["judge"] = judge(llm, q, item["gold_answer"], rec.get("answer", ""))
    rec["latency_s"] = round(time.perf_counter() - t0, 2)
    return rec


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def mean(key, pred=lambda v: bool(v)):
        vals = [pred(r[key]) for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    judged = [r["judge"] for r in rows if r.get("judge")]
    return {
        "questions": len(rows),
        "guardrail_allowed": mean("guardrail", lambda v: v == "allow"),
        "page_hit_raw_question": mean("hit_raw"),
        "page_hit_rewritten_queries": mean("hit_rewrite"),
        "page_hit_merged": mean("hit_merged"),
        "answered": mean("final_status", lambda v: v in ("answered", "partially_answered", "corrected")),
        "citations_ok": mean("citations_ok"),
        "verify_failed_then_repaired": mean("repaired"),
        "gold_figure": mean("gold_figure"),
        "judge_correct": (sum(j == "correct" for j in judged) / len(judged)) if judged else None,
        "judge_correct_or_partial": (sum(j != "incorrect" for j in judged) / len(judged)) if judged else None,
        "avg_llm_calls": statistics.mean(r["calls"] for r in rows) if rows else None,
        "p50_latency_s": statistics.median(r["latency_s"] for r in rows) if rows else None,
    }


def to_markdown(summary: Dict[str, Any]) -> str:
    lines = [f"End-to-end, prompt set {summary['prompt_set']}: {summary['questions']} FinanceBench dev questions", "", "| metric | value |", "|---|---|"]
    for k, v in summary.items():
        if k in ("prompt_set", "questions"):
            continue
        cell = "-" if v is None else (f"{v:.2f}" if k in ("avg_llm_calls", "p50_latency_s") else f"{v * 100:.1f}%")
        lines.append(f"| {k.replace('_', ' ')} | {cell} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--profile", default="medium")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--final-run", action="store_true")
    args = p.parse_args(argv)

    from app.config import settings
    from app.rag.retrieval import HybridRetriever
    from app.rag.store import PgVectorStore
    from app.services.llm_client import LLMClient

    items = select(load_items(args.data, args.final_run), args.limit)
    store = PgVectorStore()
    indexed = set()
    try:
        indexed = {c.doc_name for c in store.load_profile(args.profile, settings.embeddings_model, None)}
    except Exception as exc:
        sys.exit(f"Could not read the RAG index ({exc}). Run init-db and ingest first.")
    missing = sorted({i["doc_name"] for i in items} - indexed)
    if missing:
        print(f"Skipping {len(missing)} question document(s) not ingested yet: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")
        items = [i for i in items if i["doc_name"] in indexed]
    print(f"Running {len(items)} questions end to end (about 7-9 model calls each).", flush=True)

    rows: List[Dict[str, Any]] = []
    with LLMClient() as llm:
        retriever = HybridRetriever(store, llm, profile=args.profile, embedding_model=settings.embeddings_model,
                                    dimensions=settings.embeddings_dimensions)
        for n, item in enumerate(items, start=1):
            try:
                rows.append(run_question(llm, retriever, item, use_judge=not args.no_judge))
            except Exception as exc:
                rows.append({"question_id": item["question_id"], "error": f"{type(exc).__name__}: {exc}", "calls": 0, "latency_s": 0})
            print(f"  {n}/{len(items)}", end="\r", flush=True)

    summary = {"prompt_set": PROMPT_SET_VERSION, **summarize([r for r in rows if "error" not in r])}
    errors = [r for r in rows if "error" in r]
    print(" " * 20)
    print(to_markdown(summary))
    if errors:
        print(f"\n{len(errors)} question(s) failed with errors, e.g. {errors[0]['error'][:200]}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, default=str), encoding="utf-8")
        print(f"\nFull results: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
