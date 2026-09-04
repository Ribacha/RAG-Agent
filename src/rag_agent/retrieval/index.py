"""A small JSONL-backed cosine-similarity index.

The index is intentionally simple and inspectable.  It is appropriate for a
learning project and a modest personal knowledge base; the provider and file
format are isolated so a SQLite/FAISS/Qdrant implementation can replace it
later without changing ingestion or answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..embeddings.base import EmbeddingError, EmbeddingProvider
from ..models import Chunk
from ..storage.jsonl import read_jsonl, write_jsonl_atomic


INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk plus the metadata needed for a citation."""

    score: float
    chunk_id: str
    doc_id: str
    source_path: str
    file_type: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    heading_path: tuple[str, ...] = ()
    extraction_methods: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_chunk(cls, chunk: Mapping[str, Any], score: float) -> "SearchResult":
        return cls(
            score=float(score),
            chunk_id=str(chunk.get("chunk_id", "")),
            doc_id=str(chunk.get("doc_id", "")),
            source_path=str(chunk.get("source_path", "")),
            file_type=str(chunk.get("file_type", "")),
            text=str(chunk.get("text", "")),
            page_start=_optional_int(chunk.get("page_start")),
            page_end=_optional_int(chunk.get("page_end")),
            heading_path=tuple(str(item) for item in chunk.get("heading_path", []) or []),
            extraction_methods=tuple(
                str(item) for item in chunk.get("extraction_methods", []) or []
            ),
            warnings=tuple(str(item) for item in chunk.get("warnings", []) or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 8),
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "file_type": self.file_type,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "heading_path": list(self.heading_path),
            "extraction_methods": list(self.extraction_methods),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _IndexedRow:
    chunk: dict[str, Any]
    vector: tuple[float, ...]


class LocalVectorIndex:
    """Load and query a JSONL vector index."""

    def __init__(
        self,
        *,
        provider_name: str,
        provider_model: str,
        provider_fingerprint: str,
        dimension: int,
        rows: Sequence[_IndexedRow],
    ) -> None:
        self.provider_name = provider_name
        self.provider_model = provider_model
        self.provider_fingerprint = provider_fingerprint
        self.dimension = dimension
        self._rows = tuple(rows)

    @property
    def size(self) -> int:
        return len(self._rows)

    @classmethod
    def load(cls, path: Path) -> "LocalVectorIndex":
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"索引文件不存在：{path}")
        rows = list(read_jsonl(path))
        if not rows:
            raise ValueError(f"索引文件为空：{path}")
        meta = rows[0]
        if meta.get("_type") != "meta":
            raise ValueError("索引缺少 meta 首行，可能不是本项目生成的索引")
        if int(meta.get("schema_version", -1)) != INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"不支持的索引版本：{meta.get('schema_version')}，期望 {INDEX_SCHEMA_VERSION}"
            )
        provider_name = str(meta.get("provider_name", ""))
        provider_model = str(meta.get("provider_model", ""))
        fingerprint = str(meta.get("provider_fingerprint", ""))
        dimension = int(meta.get("dimension", 0))
        if not provider_name or not fingerprint or dimension <= 0:
            raise ValueError("索引 meta 缺少 provider 或 dimension")

        indexed: list[_IndexedRow] = []
        seen_ids: set[str] = set()
        for row in rows[1:]:
            if row.get("_type") != "chunk":
                continue
            chunk = row.get("chunk")
            vector = row.get("vector")
            if not isinstance(chunk, dict) or not isinstance(vector, list):
                raise ValueError("索引 chunk 行格式无效")
            chunk_id = str(chunk.get("chunk_id", ""))
            if not chunk_id or chunk_id in seen_ids:
                raise ValueError(f"索引存在重复或空 chunk_id：{chunk_id!r}")
            if len(vector) != dimension:
                raise ValueError(
                    f"chunk {chunk_id} 的向量维度为 {len(vector)}，期望 {dimension}"
                )
            numeric = tuple(float(value) for value in vector)
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(f"chunk {chunk_id} 的向量包含非有限数")
            seen_ids.add(chunk_id)
            indexed.append(_IndexedRow(chunk=dict(chunk), vector=numeric))
        return cls(
            provider_name=provider_name,
            provider_model=provider_model,
            provider_fingerprint=fingerprint,
            dimension=dimension,
            rows=indexed,
        )

    def search(
        self,
        query: str,
        *,
        provider: EmbeddingProvider,
        top_k: int = 5,
        min_score: float = -1.0,
        source_path: str | None = None,
        file_type: str | None = None,
    ) -> list[SearchResult]:
        """Embed a query and return deterministic top-k cosine matches."""

        if not query.strip():
            return []
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        _validate_provider(provider, self)
        vectors = provider.embed([query])
        if len(vectors) != 1:
            raise EmbeddingError("query embedding 返回数量不正确")
        query_vector = _finite_vector(vectors[0], self.dimension, "query")
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        if query_norm == 0:
            return []

        candidates: list[tuple[float, str, dict[str, Any]]] = []
        for row in self._rows:
            chunk = row.chunk
            if source_path is not None and str(chunk.get("source_path")) != source_path:
                continue
            if file_type is not None and str(chunk.get("file_type")) != file_type:
                continue
            score = _cosine(query_vector, row.vector, query_norm)
            if score >= min_score:
                candidates.append((score, str(chunk.get("chunk_id", "")), chunk))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [SearchResult.from_chunk(chunk, score) for score, _, chunk in candidates[:top_k]]


def build_vector_index(
    chunks: Iterable[Chunk | Mapping[str, Any]],
    *,
    provider: EmbeddingProvider,
    path: Path,
) -> LocalVectorIndex:
    """Embed chunks, write an inspectable index, and return the loaded index."""

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in chunks:
        chunk = item.to_dict() if isinstance(item, Chunk) else dict(item)
        chunk_id = str(chunk.get("chunk_id", ""))
        text = chunk.get("text")
        if not chunk_id or not isinstance(text, str) or not text.strip():
            raise ValueError("每个索引 chunk 必须有非空 chunk_id 和 text")
        if chunk_id in seen_ids:
            raise ValueError(f"发现重复 chunk_id：{chunk_id}")
        seen_ids.add(chunk_id)
        normalized.append(chunk)

    vectors = provider.embed([chunk["text"] for chunk in normalized])
    if len(vectors) != len(normalized):
        raise EmbeddingError(
            f"embedding 返回 {len(vectors)} 个向量，期望 {len(normalized)} 个"
        )
    declared_dimension = _provider_dimension(provider)
    if vectors:
        inferred_dimension = len(vectors[0])
        if declared_dimension is not None and declared_dimension != inferred_dimension:
            raise EmbeddingError(
                f"provider 声明维度 {declared_dimension} 与返回维度 {inferred_dimension} 不一致"
            )
        dimension = inferred_dimension
    elif declared_dimension is not None:
        # An empty corpus still needs a dimension in its metadata so a later
        # query can validate the provider configuration.
        dimension = declared_dimension
    else:
        raise EmbeddingError(
            "空索引无法从远程 embedding 推断维度，请显式提供 embedding dimension"
        )

    rows: list[dict[str, Any]] = [
        {
            "_type": "meta",
            "schema_version": INDEX_SCHEMA_VERSION,
            "provider_name": provider.name,
            "provider_model": provider.model,
            "provider_fingerprint": provider.fingerprint,
            "dimension": dimension,
            "chunk_count": len(normalized),
        }
    ]
    for chunk, vector in zip(normalized, vectors):
        numeric = _finite_vector(vector, dimension, str(chunk["chunk_id"]))
        rows.append({"_type": "chunk", "chunk": chunk, "vector": numeric})
    write_jsonl_atomic(path, rows)
    return LocalVectorIndex.load(path)


@dataclass(frozen=True)
class IndexUpdateStats:
    """统计增量索引更新实际复用和重新计算的向量数量。"""

    reused_vectors: int = 0
    embedded_vectors: int = 0


def update_vector_index(
    chunks: Iterable[Chunk | Mapping[str, Any]],
    *,
    provider: EmbeddingProvider,
    path: Path,
) -> tuple[LocalVectorIndex, IndexUpdateStats]:
    """更新 JSONL 索引，只为新增或变化的 chunk 计算 embedding。

    旧索引不存在、损坏或 provider 配置不一致时会自动退化为完整重建；这
    让 embedding 模型切换仍然安全，同时不会把旧向量混入新索引。
    """

    normalized = _normalize_chunks(chunks)
    previous: LocalVectorIndex | None = None
    if path.expanduser().resolve().exists():
        try:
            candidate = LocalVectorIndex.load(path)
            if candidate.provider_fingerprint == provider.fingerprint:
                declared_dimension = _provider_dimension(provider)
                if declared_dimension is None or declared_dimension == candidate.dimension:
                    previous = candidate
        except (OSError, ValueError, EmbeddingError):
            previous = None

    reusable: dict[str, tuple[dict[str, Any], tuple[float, ...]]] = {}
    if previous is not None:
        for row in previous._rows:
            reusable[row.chunk.get("chunk_id", "")] = (row.chunk, row.vector)

    vectors: list[list[float] | tuple[float, ...] | None] = []
    pending_texts: list[str] = []
    pending_positions: list[int] = []
    reused_count = 0
    for position, chunk in enumerate(normalized):
        old = reusable.get(str(chunk["chunk_id"]))
        if old is not None and _same_chunk_content(old[0], chunk):
            vectors.append(old[1])
            reused_count += 1
            continue
        vectors.append(None)
        pending_positions.append(position)
        pending_texts.append(chunk["text"])

    embedded = provider.embed(pending_texts)
    if len(embedded) != len(pending_texts):
        raise EmbeddingError(
            f"embedding 返回 {len(embedded)} 个向量，期望 {len(pending_texts)} 个"
        )
    for position, vector in zip(pending_positions, embedded):
        vectors[position] = vector

    dimension = _resolve_dimension(provider, vectors)
    rows: list[dict[str, Any]] = [
        {
            "_type": "meta",
            "schema_version": INDEX_SCHEMA_VERSION,
            "provider_name": provider.name,
            "provider_model": provider.model,
            "provider_fingerprint": provider.fingerprint,
            "dimension": dimension,
            "chunk_count": len(normalized),
        }
    ]
    for chunk, vector in zip(normalized, vectors):
        if vector is None:  # pragma: no cover - guarded by the length check above
            raise EmbeddingError(f"chunk {chunk['chunk_id']} 缺少 embedding")
        numeric = _finite_vector(vector, dimension, str(chunk["chunk_id"]))
        rows.append({"_type": "chunk", "chunk": chunk, "vector": numeric})
    write_jsonl_atomic(path, rows)
    return LocalVectorIndex.load(path), IndexUpdateStats(
        reused_vectors=reused_count,
        embedded_vectors=len(pending_texts),
    )


def _normalize_chunks(
    chunks: Iterable[Chunk | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in chunks:
        chunk = item.to_dict() if isinstance(item, Chunk) else dict(item)
        chunk_id = str(chunk.get("chunk_id", ""))
        text = chunk.get("text")
        if not chunk_id or not isinstance(text, str) or not text.strip():
            raise ValueError("每个索引 chunk 必须有非空 chunk_id 和 text")
        if chunk_id in seen_ids:
            raise ValueError(f"发现重复 chunk_id：{chunk_id}")
        seen_ids.add(chunk_id)
        normalized.append(chunk)
    return normalized


def _same_chunk_content(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    """只复用内容和版本元数据均未变化的向量。"""

    return (
        old.get("text") == new.get("text")
        and old.get("content_hash") == new.get("content_hash")
        and old.get("ingestion_fingerprint", "")
        == new.get("ingestion_fingerprint", "")
        and old.get("chunking_fingerprint", "")
        == new.get("chunking_fingerprint", "")
    )


def _resolve_dimension(
    provider: EmbeddingProvider,
    vectors: Sequence[Sequence[float] | None],
) -> int:
    declared_dimension = _provider_dimension(provider)
    actual_dimensions = {len(vector) for vector in vectors if vector is not None}
    if len(actual_dimensions) > 1:
        raise EmbeddingError("embedding 返回了维度不一致的向量")
    inferred_dimension = next(iter(actual_dimensions), None)
    if inferred_dimension is not None:
        if declared_dimension is not None and declared_dimension != inferred_dimension:
            raise EmbeddingError(
                f"provider 声明维度 {declared_dimension} 与返回维度 {inferred_dimension} 不一致"
            )
        return inferred_dimension
    if declared_dimension is not None:
        return declared_dimension
    raise EmbeddingError(
        "空索引无法从远程 embedding 推断维度，请显式提供 embedding dimension"
    )


def _validate_provider(provider: EmbeddingProvider, index: LocalVectorIndex) -> None:
    if provider.fingerprint != index.provider_fingerprint:
        raise EmbeddingError(
            "查询 embedding 配置与索引不一致："
            f"索引={index.provider_fingerprint}，查询={provider.fingerprint}。"
            "请使用相同 provider/model/dimension，或重新构建索引。"
        )
    dimension = _provider_dimension(provider)
    if dimension is not None and dimension != index.dimension:
        raise EmbeddingError(
            f"查询 embedding 维度 {dimension} 与索引维度 {index.dimension} 不一致"
        )


def _provider_dimension(provider: EmbeddingProvider) -> int | None:
    """Read a provider's dimension when known; remote providers may infer it lazily."""

    try:
        dimension = int(provider.dimension)
    except (AttributeError, EmbeddingError, TypeError, ValueError):
        return None
    if dimension <= 0:
        return None
    return dimension


def _finite_vector(values: Sequence[float], dimension: int, label: str) -> list[float]:
    if len(values) != dimension:
        raise EmbeddingError(
            f"{label} 向量维度为 {len(values)}，期望 {dimension}"
        )
    vector = [float(value) for value in values]
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingError(f"{label} 向量包含非有限数")
    return vector


def _cosine(left: Sequence[float], right: Sequence[float], left_norm: float) -> float:
    right_norm = math.sqrt(sum(value * value for value in right))
    if right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
