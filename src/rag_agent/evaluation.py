"""Offline retrieval evaluation for the local RAG index.

The evaluator deliberately stops at retrieval.  It does not call a chat model,
so a JSONL evaluation run is deterministic, auditable, and usable without
network access or API keys.  Labels may identify exact chunks, source paths, or
both; a result is relevant when it matches either configured label set.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .embeddings.base import EmbeddingProvider
from .retrieval.index import LocalVectorIndex, SearchResult
from .storage.jsonl import read_jsonl


@dataclass(frozen=True)
class EvaluationSample:
    """One manually labelled retrieval query."""

    query: str
    relevant_chunk_ids: tuple[str, ...] = ()
    relevant_source_paths: tuple[str, ...] = ()
    min_score: float | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        query = self.query.strip() if isinstance(self.query, str) else ""
        if not query:
            raise ValueError("评测样本 query 不能为空")
        object.__setattr__(self, "query", query)

        chunk_ids = _normalise_labels(self.relevant_chunk_ids, "relevant_chunk_ids")
        source_paths = _normalise_labels(
            self.relevant_source_paths, "relevant_source_paths"
        )
        if not chunk_ids and not source_paths:
            raise ValueError(
                "评测样本至少需要一个非空 relevant_chunk_ids 或 "
                "relevant_source_paths"
            )
        object.__setattr__(self, "relevant_chunk_ids", chunk_ids)
        object.__setattr__(self, "relevant_source_paths", source_paths)

        if self.min_score is not None:
            if isinstance(self.min_score, bool) or not isinstance(
                self.min_score, (int, float)
            ):
                raise ValueError("评测样本 min_score 必须是数字")
            score = float(self.min_score)
            if not math.isfinite(score):
                raise ValueError("评测样本 min_score 必须是有限数字")
            object.__setattr__(self, "min_score", score)

        if self.name is not None:
            name = self.name.strip() if isinstance(self.name, str) else ""
            if not name:
                raise ValueError("评测样本 name 不能为空字符串")
            object.__setattr__(self, "name", name)

    @classmethod
    def from_dict(cls, row: dict[str, Any], *, line_number: int) -> "EvaluationSample":
        """Parse one JSON object and attach its JSONL line number to errors."""

        if not isinstance(row, dict):
            raise ValueError(f"评测文件第 {line_number} 行必须是 JSON 对象")
        query = row.get("query")
        if not isinstance(query, str):
            raise ValueError(f"评测文件第 {line_number} 行 query 必须是字符串")
        chunk_ids = _read_label_list(row.get("relevant_chunk_ids"), "relevant_chunk_ids", line_number)
        source_paths = _read_label_list(
            row.get("relevant_source_paths"), "relevant_source_paths", line_number
        )
        name = row.get("name", row.get("id"))
        if name is not None and not isinstance(name, str):
            raise ValueError(f"评测文件第 {line_number} 行 name/id 必须是字符串")
        try:
            return cls(
                query=query,
                relevant_chunk_ids=tuple(chunk_ids),
                relevant_source_paths=tuple(source_paths),
                min_score=row.get("min_score"),
                name=name,
            )
        except ValueError as error:
            raise ValueError(f"评测文件第 {line_number} 行：{error}") from error


@dataclass(frozen=True)
class EvaluationSampleResult:
    """Retrieval results and metrics for one evaluation sample."""

    name: str
    query: str
    top_k: int
    min_score: float
    retrieved: tuple[SearchResult, ...]
    matched_chunk_ids: tuple[str, ...]
    matched_source_paths: tuple[str, ...]
    recall_at_k: float
    citation_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "query": self.query,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "retrieved_chunk_ids": [result.chunk_id for result in self.retrieved],
            "retrieved_source_paths": [result.source_path for result in self.retrieved],
            "matched_chunk_ids": list(self.matched_chunk_ids),
            "matched_source_paths": list(self.matched_source_paths),
            "recall_at_k": round(self.recall_at_k, 8),
            "citation_accuracy": round(self.citation_accuracy, 8),
            "results": [result.to_dict() for result in self.retrieved],
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate metrics plus per-query diagnostics."""

    top_k: int
    min_score: float
    sample_count: int
    recall_at_k: float
    citation_accuracy: float
    samples: tuple[EvaluationSampleResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "min_score": self.min_score,
            "sample_count": self.sample_count,
            "recall_at_k": round(self.recall_at_k, 8),
            "citation_accuracy": round(self.citation_accuracy, 8),
            "samples": [sample.to_dict() for sample in self.samples],
        }


def load_evaluation_samples(path: Path) -> list[EvaluationSample]:
    """Load and validate a JSONL evaluation set.

    Empty files are rejected because an aggregate score over zero queries is
    not useful and usually indicates a path or export mistake.
    """

    rows = read_jsonl(path)
    samples: list[EvaluationSample] = []
    for line_number, row in enumerate(rows, start=1):
        samples.append(EvaluationSample.from_dict(row, line_number=line_number))
    if not samples:
        raise ValueError(f"评测文件为空：{path.expanduser().resolve()}")
    return samples


def evaluate(
    index: LocalVectorIndex,
    samples: Iterable[EvaluationSample],
    *,
    provider: EmbeddingProvider,
    top_k: int = 5,
    min_score: float = 0.0,
) -> EvaluationReport:
    """Run labelled retrieval evaluation without invoking a chat model.

    ``recall_at_k`` is a hit rate: each sample contributes 1 when at least one
    relevant result appears in Top-K and 0 otherwise.  ``citation_accuracy`` is
    a retrieval proxy, calculated as the fraction of returned results matching
    a label; it is not a claim about an LLM's factual citation correctness.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        raise ValueError("min_score 必须是有限数字")
    if not math.isfinite(float(min_score)):
        raise ValueError("min_score 必须是有限数字")
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("评测样本不能为空")

    results: list[EvaluationSampleResult] = []
    for position, sample in enumerate(sample_list, start=1):
        effective_min_score = (
            sample.min_score if sample.min_score is not None else float(min_score)
        )
        retrieved = tuple(
            index.search(
                sample.query,
                provider=provider,
                top_k=top_k,
                min_score=effective_min_score,
            )
        )
        matched_chunks = tuple(
            result.chunk_id
            for result in retrieved
            if result.chunk_id in sample.relevant_chunk_ids
        )
        matched_sources = tuple(
            result.source_path
            for result in retrieved
            if result.source_path in sample.relevant_source_paths
        )
        relevant = tuple(
            result
            for result in retrieved
            if _is_relevant(result, sample)
        )
        name = sample.name or f"sample-{position}"
        results.append(
            EvaluationSampleResult(
                name=name,
                query=sample.query,
                top_k=top_k,
                min_score=effective_min_score,
                retrieved=retrieved,
                matched_chunk_ids=matched_chunks,
                matched_source_paths=matched_sources,
                recall_at_k=1.0 if relevant else 0.0,
                citation_accuracy=(len(relevant) / len(retrieved)) if retrieved else 0.0,
            )
        )

    sample_count = len(results)
    return EvaluationReport(
        top_k=top_k,
        min_score=float(min_score),
        sample_count=sample_count,
        recall_at_k=sum(result.recall_at_k for result in results) / sample_count,
        citation_accuracy=sum(result.citation_accuracy for result in results) / sample_count,
        samples=tuple(results),
    )


def _is_relevant(result: SearchResult, sample: EvaluationSample) -> bool:
    return result.chunk_id in sample.relevant_chunk_ids or result.source_path in sample.relevant_source_paths


def _read_label_list(value: Any, field_name: str, line_number: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"评测文件第 {line_number} 行 {field_name} 必须是字符串数组")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"评测文件第 {line_number} 行 {field_name} 必须是字符串数组")
    return [item for item in value]


def _normalise_labels(labels: Sequence[str], field_name: str) -> tuple[str, ...]:
    if not isinstance(labels, (tuple, list)):
        raise ValueError(f"{field_name} 必须是字符串序列")
    normalised: list[str] = []
    seen: set[str] = set()
    for value in labels:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} 必须只包含非空字符串")
        value = value.strip()
        if value not in seen:
            normalised.append(value)
            seen.add(value)
    return tuple(normalised)
