"""Build the FinanceBench dev / held-out split used to test the answer-side prompts.

Run once from the repo root (output is committed, so the split never changes):

    python -m backend.app.prompts.build_financebench_split      # or: cd backend && python -m app.prompts.build_financebench_split

* 150 FinanceBench open-source questions -> 100 dev + 50 held_out, stratified by
  question_type (metrics-generated / domain-relevant / novel-generated).
* Assignment is deterministic (SHA-256 of the question id), not random per run.
* Questions the RAG tests use (backend/tests/fixtures/rag_golden_queries.json) are
  forced into held_out, so RAG checks stay independent of prompt tuning.
* The project's 75-query golden set (Acme) is a separate file and is never read here.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data" / "reference" / "financebench" / "data"
FIXTURE = ROOT / "backend" / "tests" / "fixtures" / "rag_golden_queries.json"
OUT = ROOT / "data" / "evaluation"
DEV_SHARE = 2 / 3

_NUM = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?\s?%?")


def numeric_gold(answer: str) -> List[float]:
    """Numbers in a short numeric gold answer ("$1577.00", "0.4%", "1.33"); [] for prose answers."""
    text = answer.strip()
    if len(text) > 40:
        return []
    out: List[float] = []
    for m in _NUM.finditer(text):
        tok = m.group(0).strip()
        neg = tok.startswith("(") and tok.endswith(")") or tok.startswith("-")
        digits = re.sub(r"[^\d.]", "", tok)
        if not digits or digits == ".":
            continue
        val = float(digits)
        out.append(-val if neg else val)
    return out


def _bucket(qid: str) -> float:
    return int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def build() -> Dict[str, int]:
    docs = {d["doc_name"]: d for d in (json.loads(l) for l in open(SRC / "financebench_document_information.jsonl", encoding="utf-8"))}
    rows = [json.loads(l) for l in open(SRC / "financebench_open_source.jsonl", encoding="utf-8")]

    forced_holdout = set()
    if FIXTURE.exists():
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        forced_holdout |= {f["source_question_id"] for f in fixture if f.get("source_question_id")}
        fixture_docs = {f["doc_name"] for f in fixture}
        forced_holdout |= {r["financebench_id"] for r in rows if r["doc_name"] in fixture_docs and "capital expenditure" in r["question"].lower()}

    records: List[Dict[str, Any]] = []
    for r in rows:
        meta = docs.get(r["doc_name"], {})
        evidence = []
        for i, e in enumerate(r["evidence"], start=1):
            page_index = e.get("evidence_page_num")
            evidence.append({
                "source_id": f"S{i}",
                "document_name": e.get("doc_name") or r["doc_name"],
                "page": page_index + 1 if isinstance(page_index, int) else None,  # FinanceBench page numbers are 0-based
                "period": f"FY{meta['doc_period']}" if meta.get("doc_period") else "",
                "text": e.get("evidence_text", ""),
            })
        records.append({
            "question_id": r["financebench_id"],
            "company": r["company"],
            "doc_name": r["doc_name"],
            "doc_type": meta.get("doc_type"),
            "question_type": r["question_type"],
            "question_reasoning": r.get("question_reasoning"),
            "question": r["question"],
            "gold_answer": r["answer"],
            "justification": r.get("justification"),
            "numeric_gold": numeric_gold(r["answer"]),
            "evidence": evidence,
        })

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_type[rec["question_type"]].append(rec)

    dev, held = [], []
    for qtype, recs in sorted(by_type.items()):
        recs.sort(key=lambda r: _bucket(r["question_id"]))
        n_dev = round(len(recs) * DEV_SHARE)
        forced = [r for r in recs if r["question_id"] in forced_holdout]
        free = [r for r in recs if r["question_id"] not in forced_holdout]
        dev += free[:n_dev]
        held += free[n_dev:] + forced

    for r in dev:
        r["split"] = "dev"
    for r in held:
        r["split"] = "held_out"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("financebench_dev.jsonl", dev), ("financebench_holdout.jsonl", held)):
        with open(OUT / name, "w", encoding="utf-8", newline="\n") as fh:
            for r in sorted(part, key=lambda r: r["question_id"]):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"dev": len(dev), "held_out": len(held), "forced_holdout": len(forced_holdout)}


if __name__ == "__main__":
    print(build())
