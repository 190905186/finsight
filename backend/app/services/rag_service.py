from __future__ import annotations

import re
from typing import Any, Dict, List

from app.services.sample_data import RAG_DOCS


class RAGService:
    def __init__(self) -> None:
        self.documents = RAG_DOCS

    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        tokens = {token.lower() for token in re.findall(r"[a-zA-Z0-9]+", query)}
        scored: List[tuple[float, Dict[str, Any]]] = []
        for doc in self.documents:
            text = doc["text"].lower()
            score = 0.0
            for token in tokens:
                score += text.count(token) * 2.0
            if "revenue" in text and "revenue" in query.lower():
                score += 3.0
            if "hotel" in text and "hotel" in query.lower():
                score += 3.0
            if "vendor" in text and "vendor" in query.lower():
                score += 3.0
            scored.append((score, doc))
        top = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
        return [{"document": doc["title"], "section": doc["section"], "page": doc["page"], "snippet": doc["text"], "score": round(score, 2)} for score, doc in top]
