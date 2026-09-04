"""A deterministic, dependency-free embedding provider.

This is deliberately not presented as a replacement for a trained embedding
model.  It uses hashed word and character features, which makes the complete
RAG path runnable offline and gives us a stable baseline for tests.  Later we
can swap it for a hosted or local neural model through the same provider
interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
import unicodedata
from typing import Sequence

from .base import EmbeddingError


HASH_EMBEDDING_VERSION = "hash-v1"
_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


@dataclass(frozen=True)
class HashEmbeddingProvider:
    """Feature-hashing embeddings suitable for a small local index."""

    dimension: int = 384

    def __post_init__(self) -> None:
        if self.dimension < 32:
            raise ValueError("embedding dimension 至少为 32")

    @property
    def name(self) -> str:
        return "hash"

    @property
    def model(self) -> str:
        return HASH_EMBEDDING_VERSION

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
        normalized = unicodedata.normalize("NFKC", text).lower()
        vector = [0.0] * self.dimension
        features = list(_features(normalized))
        if not features:
            return vector

        # A signed hash reduces the effect of collisions while keeping the
        # implementation independent of numpy and platform hash randomization.
        for feature, weight in features:
            digest = hashlib.blake2b(
                feature.encode("utf-8"), digest_size=8
            ).digest()
            value = int.from_bytes(digest, "big", signed=False)
            index = value % self.dimension
            sign = 1.0 if ((value >> 17) & 1) else -1.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


def _features(text: str) -> list[tuple[str, float]]:
    """Return weighted lexical and character features for one string."""

    features: list[tuple[str, float]] = []
    words = _WORD_RE.findall(text)
    for word in words:
        features.append((f"w:{word}", 1.0))
        # Character n-grams help Chinese queries and small spelling variants.
        chars = list(word)
        if len(chars) > 1:
            for size in (2, 3):
                for start in range(len(chars) - size + 1):
                    features.append((f"c{size}:{''.join(chars[start:start + size])}", 0.65))

    cjk = _CJK_RE.findall(text)
    if len(cjk) > 1:
        for start in range(len(cjk) - 1):
            features.append((f"b:{cjk[start]}{cjk[start + 1]}", 0.9))
    return features
