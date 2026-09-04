"""Construct embedding providers from CLI/environment-friendly settings."""

from __future__ import annotations

import os

from .base import EmbeddingError, EmbeddingProvider
from .chinese import ChineseNgramEmbeddingProvider
from .hashing import HashEmbeddingProvider
from .openai_compatible import OpenAICompatibleEmbeddingProvider


def create_embedding_provider(
    name: str | None = None,
    *,
    dimension: int | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> EmbeddingProvider:
    """Create a provider without importing optional SDKs unnecessarily."""

    selected = (name or os.getenv("EMBEDDING_PROVIDER") or "hash").strip().lower()
    if selected in {"hash", "local"}:
        return HashEmbeddingProvider(dimension=dimension or 384)
    if selected in {"chinese", "zh", "zh-hash", "ngram"}:
        return ChineseNgramEmbeddingProvider(dimension=dimension or 512)
    if selected in {"openai", "openai-compatible", "remote"}:
        key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")
        if not key:
            raise EmbeddingError(
                "远程 embedding 缺少 API Key；请设置 EMBEDDING_API_KEY，"
                "或显式使用 --embedding-provider hash"
            )
        return OpenAICompatibleEmbeddingProvider(
            api_key=key,
            model=model or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small",
            base_url=base_url or os.getenv("EMBEDDING_BASE_URL"),
            _dimension=dimension,
        )
    raise EmbeddingError(f"未知 embedding provider：{selected}")
