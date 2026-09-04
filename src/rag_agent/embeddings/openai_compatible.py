"""Optional OpenAI-compatible embedding provider.

The import is lazy so the default offline installation does not require the
OpenAI SDK.  ``base_url`` can point to an OpenAI-compatible service, but the
service must actually implement ``/embeddings``; chat compatibility alone is
not enough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .base import EmbeddingError


@dataclass
class OpenAICompatibleEmbeddingProvider:
    api_key: str
    model: str
    base_url: str | None = None
    _client: object | None = None
    _dimension: int | None = None

    def __post_init__(self) -> None:
        if self._dimension is not None and self._dimension <= 0:
            raise ValueError("embedding dimension 必须大于 0")

    @property
    def name(self) -> str:
        return "openai-compatible"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise EmbeddingError(
                "远程 embedding 的维度未知；请提供 --embedding-dimension，"
                "或先调用 embed()"
            )
        return self._dimension

    @property
    def fingerprint(self) -> str:
        endpoint = (self.base_url or "default").rstrip("/")
        return f"{self.name}:{endpoint}:{self.model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        try:
            response = client.embeddings.create(model=self.model, input=list(texts))
            data = sorted(response.data, key=lambda item: item.index)
            vectors = [list(map(float, item.embedding)) for item in data]
        except Exception as error:  # SDK/provider exception types vary.
            raise EmbeddingError(f"远程 embedding 请求失败：{error}") from error

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"远程 embedding 返回 {len(vectors)} 个向量，期望 {len(texts)} 个"
            )
        dimension = len(vectors[0]) if vectors else 0
        if dimension <= 0 or any(len(vector) != dimension for vector in vectors):
            raise EmbeddingError("远程 embedding 返回了维度不一致的向量")
        self._dimension = dimension
        return vectors

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ModuleNotFoundError as error:
            raise EmbeddingError(
                "远程 embedding 需要 OpenAI SDK，请安装 `python -m pip install '.[llm]'`"
            ) from error
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client
