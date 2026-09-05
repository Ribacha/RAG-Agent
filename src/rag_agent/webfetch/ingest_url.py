"""网页内容的导入整合：合并新页面与既有索引数据。

合并策略与 ``ingest --incremental`` 语义一致：

- 抓到的 URL：新 chunks 直接替换旧 chunks（内容没变的 URL 生成的
  chunk_id 相同，``update_vector_index`` 会自动复用旧向量，不重复调用
  embedding）；
- 没抓到的 URL：既有 chunks/manifest 原样保留——重新爬取某个入口不应
  隐式删除之前导入的其他页面；想清理请重建索引。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..chunking.splitter import ChunkConfig, chunk_document
from ..models import DocumentRecord


@dataclass(frozen=True)
class WebMergeCounts:
    """每个 URL 归入"新增/更新/未变化"哪一类的统计。"""

    added: int = 0
    updated: int = 0
    unchanged: int = 0

    @property
    def total(self) -> int:
        return self.added + self.updated + self.unchanged


@dataclass(frozen=True)
class WebMergeResult:
    """合并后的完整数据集与统计。"""

    chunks: list[dict[str, object]]
    manifests: list[dict[str, object]]
    counts: WebMergeCounts


def merge_web_documents(
    records: Sequence[DocumentRecord],
    existing_chunks: Sequence[Mapping[str, object]],
    existing_manifests: Sequence[Mapping[str, object]],
    *,
    chunk_config: ChunkConfig,
) -> WebMergeResult:
    """把爬取到的文档与既有 chunks/manifests 合并成一份完整快照。"""

    crawled_urls = {record.source_path for record in records}

    # 只保留"这次没抓到的页面"的旧 chunks；被抓到的页面用新结果替换。
    kept_chunks = [
        dict(chunk)
        for chunk in existing_chunks
        if chunk.get("source_path") not in crawled_urls
    ]

    new_manifest_by_url: dict[str, dict[str, object]] = {}
    kept_manifests = [
        dict(row)
        for row in existing_manifests
        if row.get("source_path") not in crawled_urls
    ]

    chunks = list(kept_chunks)
    added = updated = unchanged = 0
    for record in records:
        document_chunks = chunk_document(record, config=chunk_config)
        chunks.extend(chunk.to_dict() for chunk in document_chunks)

        manifest = record.to_manifest_dict(len(document_chunks))
        new_manifest_by_url[record.source_path] = manifest

        previous = next(
            (row for row in existing_manifests if row.get("source_path") == record.source_path),
            None,
        )
        if previous is None:
            added += 1
        elif (
            previous.get("content_hash") == record.content_hash
            and previous.get("ingestion_fingerprint") == record.ingestion_fingerprint
        ):
            unchanged += 1
        else:
            updated += 1

    manifests = kept_manifests + list(new_manifest_by_url.values())
    return WebMergeResult(
        chunks=chunks,
        manifests=manifests,
        counts=WebMergeCounts(added=added, updated=updated, unchanged=unchanged),
    )
