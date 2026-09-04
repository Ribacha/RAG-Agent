"""增量导入协调：复用未变化来源，只重建发生变化的文档。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..chunking.splitter import ChunkConfig, chunk_document
from ..models import (
    FailedDocument,
    IngestionError,
    config_fingerprint,
    sha256_bytes,
)
from .pdf import PdfOptions
from .pipeline import SUPPORTED_SUFFIXES, discover_files, load_file
from .text import read_text


@dataclass(frozen=True)
class IncrementalIngestionResult:
    chunks: tuple[dict[str, Any], ...]
    manifests: tuple[dict[str, Any], ...]
    failures: tuple[FailedDocument, ...]
    documents_added: int
    documents_updated: int
    documents_unchanged: int
    documents_deleted: int
    documents_failed: int
    chunks_reused: int
    chunks_generated: int


def incremental_ingest(
    input_path: Path,
    *,
    existing_chunks: Iterable[Mapping[str, Any]] = (),
    existing_manifests: Iterable[Mapping[str, Any]] = (),
    chunk_config: ChunkConfig | None = None,
    pdf_options: PdfOptions | None = None,
    max_file_bytes: int = 50 * 1024 * 1024,
) -> IncrementalIngestionResult:
    """合并指定输入范围和旧快照，得到下一版 chunks/manifest。

    输入为目录时，目录内已从磁盘删除的来源会从快照移除；输入为单文件时只
    更新该文件，不会影响索引里的其他来源。解析失败时保留该来源的上一版结果。
    """

    input_path = input_path.expanduser().resolve()
    files = discover_files(input_path, allow_empty=input_path.is_dir())
    config = chunk_config or ChunkConfig()
    old_chunks = [dict(row) for row in existing_chunks]
    old_manifests = [dict(row) for row in existing_manifests]
    chunks_by_source = _group_by_source(old_chunks)
    manifest_by_source = {
        _source_key(row): row for row in old_manifests if _source_key(row)
    }

    current_paths = {str(path) for path in files}
    scoped_old_sources = {
        source
        for source in set(chunks_by_source) | set(manifest_by_source)
        if _is_in_scope(source, input_path)
    }
    deleted_sources = scoped_old_sources - current_paths if input_path.is_dir() else set()

    kept_sources = (set(chunks_by_source) | set(manifest_by_source)) - deleted_sources
    final_chunks_by_source = {
        source: list(chunks_by_source.get(source, ())) for source in kept_sources
    }
    final_manifests = {
        source: dict(manifest_by_source[source])
        for source in kept_sources
        if source in manifest_by_source
    }

    added = 0
    updated = 0
    unchanged = 0
    chunks_reused = 0
    chunks_generated = 0
    failures: list[FailedDocument] = []
    expected_chunking_fingerprint = config.fingerprint()

    for path in files:
        source = str(path)
        old_manifest = manifest_by_source.get(source)
        try:
            size_bytes = path.stat().st_size
            if size_bytes > max_file_bytes:
                raise IngestionError(
                    f"文件大小超过上限 {max_file_bytes} 字节：{path.name}"
                )
            content_hash = sha256_bytes(path.read_bytes())
            expected_ingestion_fingerprint = _expected_ingestion_fingerprint(
                path, pdf_options=pdf_options
            )
            if _can_reuse(
                old_manifest,
                content_hash=content_hash,
                ingestion_fingerprint=expected_ingestion_fingerprint,
                chunking_fingerprint=expected_chunking_fingerprint,
                old_chunks=chunks_by_source.get(source, ()),
            ):
                unchanged += 1
                chunks_reused += len(chunks_by_source.get(source, ()))
                continue

            document = load_file(path, pdf_options=pdf_options)
            document_chunks = [
                chunk.to_dict() for chunk in chunk_document(document, config=config)
            ]
            final_chunks_by_source[source] = document_chunks
            final_manifests[source] = document.to_manifest_dict(len(document_chunks))
            chunks_generated += len(document_chunks)
            if old_manifest is None:
                added += 1
            else:
                updated += 1
        except Exception as error:
            failures.append(
                FailedDocument(
                    source_path=source,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
            # A failed refresh must not discard the last known-good version.
            if source not in final_chunks_by_source:
                final_chunks_by_source.pop(source, None)
                final_manifests.pop(source, None)

    chunks = tuple(
        row
        for source in sorted(final_chunks_by_source, key=str.casefold)
        for row in sorted(
            final_chunks_by_source[source],
            key=lambda item: (int(item.get("chunk_index", 0)), str(item.get("chunk_id", ""))),
        )
    )
    manifests = tuple(
        final_manifests[source]
        for source in sorted(final_manifests, key=str.casefold)
    )
    return IncrementalIngestionResult(
        chunks=chunks,
        manifests=manifests,
        failures=tuple(failures),
        documents_added=added,
        documents_updated=updated,
        documents_unchanged=unchanged,
        documents_deleted=len(deleted_sources),
        documents_failed=len(failures),
        chunks_reused=chunks_reused,
        chunks_generated=chunks_generated,
    )


def _group_by_source(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source = _source_key(row)
        if source:
            grouped.setdefault(source, []).append(dict(row))
    return grouped


def _source_key(row: Mapping[str, Any]) -> str:
    value = row.get("source_path")
    if not isinstance(value, str) or not value.strip():
        return ""
    return str(Path(value).expanduser().resolve(strict=False))


def _is_in_scope(source: str, input_path: Path) -> bool:
    source_path = Path(source).expanduser().resolve(strict=False)
    if input_path.is_file():
        return source_path == input_path
    return source_path == input_path or input_path in source_path.parents


def _can_reuse(
    manifest: Mapping[str, Any] | None,
    *,
    content_hash: str,
    ingestion_fingerprint: str,
    chunking_fingerprint: str,
    old_chunks: Iterable[Mapping[str, Any]],
) -> bool:
    if manifest is None:
        return False
    chunks = list(old_chunks)
    return (
        manifest.get("content_hash") == content_hash
        and manifest.get("ingestion_fingerprint") == ingestion_fingerprint
        and manifest.get("chunking_fingerprint") == chunking_fingerprint
        and int(manifest.get("chunk_count", -1)) == len(chunks)
    )


def _expected_ingestion_fingerprint(
    path: Path,
    *,
    pdf_options: PdfOptions | None,
) -> str:
    """在不做完整解析的情况下计算当前文件的解析配置指纹。"""

    file_type = SUPPORTED_SUFFIXES[path.suffix.lower()]
    if file_type == "pdf":
        return (pdf_options or PdfOptions()).fingerprint()
    result = read_text(path)
    if file_type == "markdown":
        return config_fingerprint(
            "ingestion",
            {
                "file_type": "markdown",
                "normalization": "crlf_to_lf",
                "encoding": result.encoding,
                "heading_parser": "atx-v1",
            },
        )
    return config_fingerprint(
        "ingestion",
        {
            "file_type": file_type,
            "normalization": "crlf_to_lf",
            "encoding": result.encoding,
        },
    )
