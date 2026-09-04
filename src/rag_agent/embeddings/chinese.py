"""Dependency-free Chinese-oriented lexical embeddings.

This provider is an offline baseline, not a neural semantic model.  It uses
overlapping Chinese 1-4 character n-grams plus stable Latin/digit tokens, so
Chinese phrase and punctuation variants share more features while remaining
deterministic and inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import unicodedata
from typing import Sequence

from .base import EmbeddingError


CHINESE_EMBEDDING_VERSION = "chinese-ngram-v1"
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._:+/-][a-z0-9]+)*", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


@dataclass(frozen=True)
class ChineseNgramEmbeddingProvider:
    """Stable character n-gram vectors for Chinese-heavy corpora."""

    dimension: int = 512

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise ValueError("embedding dimension 必须是整数")
        if self.dimension < 64:
            raise ValueError("chinese embedding dimension 至少为 64")

    @property
    def name(self) -> str:
        return "chinese"

    @property
    def model(self) -> str:
        return CHINESE_EMBEDDING_VERSION

    @property
    def fingerprint(self) -> str:
        return f"{self.name}:{self.model}:d{self.dimension}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if not isinstance(text, str):
                raise EmbeddingError("embedding 输入必须是字符串")
            vectors.append(self._embed_one(text))
        return vectors

    def _embed_one(self, text: str) -> list[float]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        vector = [0.0] * self.dimension
        features = _features(normalized)
        if not features:
            return vector
        for feature, weight in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big", signed=False)
            index = value % self.dimension
            sign = 1.0 if ((value >> 19) & 1) else -1.0
            vector[index] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


def _features(text: str) -> list[tuple[str, float]]:
    """Extract weighted CJK n-grams and stable non-CJK lexical features."""

    features: list[tuple[str, float]] = []
    for run in _CJK_RUN_RE.findall(text):
        for size, weight in ((1, 0.35), (2, 1.0), (3, 0.9), (4, 0.7)):
            if len(run) < size:
                continue
            for start in range(len(run) - size + 1):
                features.append((f"cjk{size}:{run[start:start + size]}", weight))
    for token in _TOKEN_RE.findall(text):
        features.append((f"token:{token}", 1.1))
        for size, weight in ((2, 0.45), (3, 0.3)):
            if len(token) < size:
                continue
            for start in range(len(token) - size + 1):
                features.append((f"latin{size}:{token[start:start + size]}", weight))
    if _CJK_RE.search(text) is None and not _TOKEN_RE.search(text):
        compact = "".join(char for char in text if not char.isspace())
        if compact:
            features.append((f"symbol:{compact}", 0.25))
    return features
