from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from app.config import settings

try:
    import redis
except Exception:  # noqa: BLE001
    redis = None  # type: ignore


class CacheRateLimiter:
    def __init__(self) -> None:
        self._memory_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._memory_hits: dict[str, list[float]] = {}
        self._client = None
        if redis is not None:
            try:
                self._client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
                self._client.ping()
            except Exception:  # noqa: BLE001
                self._client = None

    def _cache_key(self, question: str, company: str | None, index_version: str) -> str:
        raw = json.dumps(
            {"q": question.strip().lower(), "company": company or "", "index": index_version},
            sort_keys=True,
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"finsight:cache:{digest}"

    def get_cached(self, question: str, company: str | None, index_version: str) -> dict[str, Any] | None:
        key = self._cache_key(question, company, index_version)
        if self._client is not None:
            payload = self._client.get(key)
            if payload:
                return json.loads(payload)
            return None
        item = self._memory_cache.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at < time.time():
            self._memory_cache.pop(key, None)
            return None
        return value

    def set_cached(
        self,
        question: str,
        company: str | None,
        index_version: str,
        payload: dict[str, Any],
        ttl_seconds: int = 300,
    ) -> None:
        key = self._cache_key(question, company, index_version)
        if self._client is not None:
            self._client.setex(key, ttl_seconds, json.dumps(payload))
            return
        self._memory_cache[key] = (time.time() + ttl_seconds, payload)

    def allow(self, identity: str) -> bool:
        now = time.time()
        window = 60.0
        limit = settings.rate_limit_per_minute
        key = f"finsight:rl:{identity}"
        if self._client is not None:
            pipe = self._client.pipeline()
            pipe.zremrangebyscore(key, 0, now - window)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, int(window) + 1)
            _, _, count, _ = pipe.execute()
            return int(count) <= limit
        hits = [ts for ts in self._memory_hits.get(key, []) if ts >= now - window]
        hits.append(now)
        self._memory_hits[key] = hits
        return len(hits) <= limit
