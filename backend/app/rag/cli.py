from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.config import settings
from app.evaluation import load_golden_questions
from app.rag.chunking import CHUNKING_PROFILES
from app.rag.evaluation import (
    RETRIEVAL_MODES,
    RetrievalQuestion,
    evaluate_answers,
    evaluate_retrieval,
    load_financebench_questions,
)
from app.rag.embeddings import embed_texts
from app.rag.ingestion import (
    PDFIngestor,
    configured_profiles,
    load_configured_document_metadata,
)
from app.rag.pipeline import RAGPipeline
from app.rag.retrieval import HybridRetriever
from app.rag.store import PgVectorStore
from app.services.llm_client import LLMClient


def build_parser() -> argparse.ArgumentParser:
    profile_names = [profile.name for profile in CHUNKING_PROFILES]
    parser = argparse.ArgumentParser(description="Index and evaluate the FinanceBench RAG corpus.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init-db", help="Enable pgvector and create the RAG tables")

    ingest = commands.add_parser("ingest", help="Extract, chunk, and embed PDF documents")
    ingest.add_argument("--profile", choices=profile_names, default="medium")
    ingest.add_argument("--pdf-dir", type=Path, default=settings.financebench_pdf_dir)

    evaluate = commands.add_parser(
        "evaluate", help="Measure page-level retrieval recall without answer generation"
    )
    evaluate.add_argument("--profile", choices=profile_names, default="medium")
    evaluate.add_argument("--mode", choices=RETRIEVAL_MODES, default="sharedStore")
    evaluate.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    evaluate.add_argument(
        "--dataset", choices=("golden", "financebench"), default="golden"
    )

    experiment = commands.add_parser(
        "experiment", help="Compare chunk profiles using retrieval-only metrics"
    )
    experiment.add_argument(
        "--profiles", nargs="+", choices=profile_names, default=profile_names
    )
    experiment.add_argument("--mode", choices=RETRIEVAL_MODES, default="sharedStore")
    experiment.add_argument("--pdf-dir", type=Path, default=settings.financebench_pdf_dir)
    experiment.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    experiment.add_argument(
        "--dataset", choices=("golden", "financebench"), default="golden"
    )

    answer_evaluate = commands.add_parser(
        "answer-evaluate",
        help="Generate answers and score them separately from retrieval metrics",
    )
    answer_evaluate.add_argument("--profile", choices=profile_names, default="medium")
    answer_evaluate.add_argument("--mode", choices=RETRIEVAL_MODES, default="sharedStore")
    answer_evaluate.add_argument("--top-k", type=int, default=5)
    answer_evaluate.add_argument("--limit", type=int)
    answer_evaluate.add_argument("--output", type=Path, required=True)
    answer_evaluate.add_argument(
        "--dataset", choices=("golden", "financebench"), default="golden"
    )
    compare_prompts = commands.add_parser(
        "compare-prompts",
        help="Compare zero-shot, few-shot, and private-CoT on a gold split",
    )
    compare_prompts.add_argument("--profile", choices=profile_names, default="medium")
    compare_prompts.add_argument("--mode", choices=RETRIEVAL_MODES, default="sharedStore")
    compare_prompts.add_argument("--top-k", type=int, default=5)
    compare_prompts.add_argument(
        "--split", choices=("prompt_dev", "prompt_holdout"), default="prompt_dev"
    )
    compare_prompts.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_evaluation_questions(dataset: str) -> list[RetrievalQuestion]:
    if dataset == "golden":
        return load_golden_questions()
    if dataset == "financebench":
        return load_financebench_questions()
    raise ValueError(f"Unsupported evaluation dataset: {dataset}")


def _evaluate_profile(
    store: PgVectorStore,
    client: LLMClient,
    profile: str,
    k_values: Sequence[int],
    questions: Sequence[RetrievalQuestion],
    retrieval_mode: str = "sharedStore",
    query_embeddings: Sequence[Sequence[float]] | None = None,
) -> dict:
    retriever = HybridRetriever(store, client, profile=profile)
    return evaluate_retrieval(
        questions,
        retriever,
        client,
        k_values=k_values,
        retrieval_mode=retrieval_mode,
        query_embeddings=query_embeddings,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    store = PgVectorStore()
    try:
        store.initialize()
        if args.command == "init-db":
            print("pgvector extension and RAG tables are ready.")
            return

        if args.command == "ingest":
            profile = configured_profiles([args.profile])[0]
            with LLMClient() as client:
                counts = PDFIngestor(store, client).ingest_directory(
                    args.pdf_dir,
                    profile,
                    load_configured_document_metadata(),
                )
            print(
                json.dumps(
                    {
                        "profile": profile.name,
                        "documents_indexed": len(counts),
                        "chunks_indexed": sum(counts.values()),
                    },
                    indent=2,
                )
            )
            return

        if args.command == "evaluate":
            questions = _load_evaluation_questions(args.dataset)
            with LLMClient() as client:
                metrics = _evaluate_profile(
                    store, client, args.profile, args.k, questions, args.mode
                )
            print(json.dumps(metrics, indent=2))
            return

        if args.command == "experiment":
            profiles = configured_profiles(args.profiles)
            metadata = load_configured_document_metadata()
            questions = _load_evaluation_questions(args.dataset)
            results = []
            with LLMClient() as client:
                query_embeddings = embed_texts(
                    client,
                    [question.query for question in questions],
                    model=settings.embeddings_model,
                    dimensions=settings.embeddings_dimensions,
                    batch_size=settings.rag_embedding_batch_size,
                )
                ingestor = PDFIngestor(store, client)
                for profile in profiles:
                    counts = ingestor.ingest_directory(
                        args.pdf_dir, profile, metadata
                    )
                    metrics = _evaluate_profile(
                        store,
                        client,
                        profile.name,
                        args.k,
                        questions,
                        args.mode,
                        query_embeddings,
                    )
                    metrics["documents_indexed"] = len(counts)
                    metrics["chunks_indexed"] = sum(counts.values())
                    results.append(metrics)
            print(json.dumps(results, indent=2))
            return

        if args.command == "answer-evaluate":
            questions = _load_evaluation_questions(args.dataset)
            pipeline = RAGPipeline(store=store, profile=args.profile)
            metrics = evaluate_answers(
                questions,
                pipeline,
                retrieval_mode=args.mode,
                top_k=args.top_k,
                limit=args.limit,
            )
            result_rows = metrics.pop("results")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
                for row in result_rows:
                    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            metrics["results_path"] = str(args.output.resolve())
            print(json.dumps(metrics, indent=2))
            return

        if args.command == "compare-prompts":
            questions = load_golden_questions(args.split)
            pipeline = RAGPipeline(store=store, profile=args.profile)
            results = []
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for strategy in ("zero_shot", "few_shot", "cot"):
                metrics = evaluate_answers(
                    questions,
                    pipeline,
                    retrieval_mode=args.mode,
                    top_k=args.top_k,
                    prompting_strategy=strategy,
                )
                result_rows = metrics.pop("results")
                output_path = args.output_dir / f"{strategy}.jsonl"
                with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
                    for row in result_rows:
                        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                metrics["results_path"] = str(output_path.resolve())
                results.append(metrics)
            print(json.dumps(results, indent=2))
    finally:
        store.close()


if __name__ == "__main__":
    main()
