"""数据契约。

导入器、分块器、向量索引和回答 Agent 都通过这里的对象交接数据。把契约先
固定下来，可以在以后替换 PDF/OCR/Embedding 实现时不牵动上层代码。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


INGESTION_VERSION = "0.2"
CHUNKING_VERSION = "0.2"


def canonical_source_path(source_path: str | Path) -> str:
    """返回用于身份计算的规范化路径。

    `source_path` 仍然作为展示/引用字段保存；身份计算单独使用规范化后的
    绝对路径，避免相对路径和 `..` 造成同一文件出现多个 source_id。
    """

    return str(Path(source_path).expanduser().resolve(strict=False))


def source_id_for_path(source_path: str | Path) -> str:
    """根据路径生成稳定的来源身份（与文件内容解耦）。"""

    return sha256_text(f"source:v1:{canonical_source_path(source_path)}")


def document_id_for_source(source_id: str, content_hash: str) -> str:
    """组合来源身份和原始文件哈希，生成文档版本身份。"""

    return sha256_text(f"document:v1:{source_id}|{content_hash}")


def config_fingerprint(namespace: str, config: Any) -> str:
    """对解析/分块配置做确定性指纹，避免配置变化误复用旧索引。"""

    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256_text(f"{namespace}:v1:{payload}")


def sha256_bytes(data: bytes) -> str:
    """返回文件内容的稳定 SHA-256 哈希。"""

    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """返回文本内容的稳定 SHA-256 哈希。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TextBlock:
    """解析器输出的最小文本单元。

    Markdown/TXT 通常按段落产生 block，PDF 按页产生 block。空 block 也可以
    保留，用于记录“这一页需要 OCR 但 OCR 不可用”等诊断信息；分块阶段会跳过
    空文本。
    """

    doc_id: str
    source_path: str
    file_type: str
    text: str
    extraction_method: str
    page_number: int | None = None
    heading_path: tuple[str, ...] = ()
    # Deprecated compatibility aliases.  Both names mean offsets in the
    # normalized text supplied to the chunker, never byte offsets in the file.
    source_char_start: int | None = None
    source_char_end: int | None = None
    warnings: tuple[str, ...] = ()
    # New canonical identity field.  It is optional so older callers that only
    # provide doc_id/source_path continue to work.
    source_id: str | None = None
    # Canonical offsets are measured in the normalized text passed to the
    # chunker, not raw file bytes.  The old source_char_* names remain aliases.
    normalized_char_start: int | None = None
    normalized_char_end: int | None = None
    encoding: str | None = None

    def __post_init__(self) -> None:
        if self.source_id is None:
            object.__setattr__(self, "source_id", source_id_for_path(self.source_path))
        normalized_start = (
            self.normalized_char_start
            if self.normalized_char_start is not None
            else self.source_char_start
        )
        normalized_end = (
            self.normalized_char_end
            if self.normalized_char_end is not None
            else self.source_char_end
        )
        object.__setattr__(self, "normalized_char_start", normalized_start)
        object.__setattr__(self, "normalized_char_end", normalized_end)
        object.__setattr__(self, "source_char_start", normalized_start)
        object.__setattr__(self, "source_char_end", normalized_end)

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON 的字典。"""

        value = asdict(self)
        value["heading_path"] = list(self.heading_path)
        value["warnings"] = list(self.warnings)
        return value


@dataclass(frozen=True)
class Chunk:
    """供后续 Embedding/检索使用的文本块。"""

    chunk_id: str
    doc_id: str
    source_path: str
    file_type: str
    text: str
    chunk_index: int
    content_hash: str
    page_start: int | None = None
    page_end: int | None = None
    heading_path: tuple[str, ...] = ()
    char_start: int | None = None
    char_end: int | None = None
    extraction_methods: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    ingestion_version: str = INGESTION_VERSION
    chunking_version: str = CHUNKING_VERSION
    source_id: str | None = None
    ingestion_fingerprint: str = ""
    chunking_fingerprint: str = ""
    normalized_char_start: int | None = None
    normalized_char_end: int | None = None

    def __post_init__(self) -> None:
        if self.source_id is None:
            object.__setattr__(self, "source_id", source_id_for_path(self.source_path))
        normalized_start = (
            self.normalized_char_start
            if self.normalized_char_start is not None
            else self.char_start
        )
        normalized_end = (
            self.normalized_char_end
            if self.normalized_char_end is not None
            else self.char_end
        )
        object.__setattr__(self, "normalized_char_start", normalized_start)
        object.__setattr__(self, "normalized_char_end", normalized_end)
        object.__setattr__(self, "char_start", normalized_start)
        object.__setattr__(self, "char_end", normalized_end)

    def to_dict(self) -> dict[str, Any]:
        """转换为可写入 JSON 的字典。"""

        value = asdict(self)
        value["heading_path"] = list(self.heading_path)
        value["extraction_methods"] = list(self.extraction_methods)
        value["warnings"] = list(self.warnings)
        return value


@dataclass
class DocumentRecord:
    """一个文件的解析结果和摘要信息。"""

    doc_id: str
    source_path: str
    file_type: str
    content_hash: str
    size_bytes: int
    blocks: list[TextBlock] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_id: str | None = None
    ingestion_fingerprint: str = ""
    chunking_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.source_id is None:
            self.source_id = source_id_for_path(self.source_path)
        # Old code used the raw content hash as doc_id.  Upgrade that specific
        # shape automatically while preserving arbitrary caller-supplied IDs.
        if self.doc_id == self.content_hash:
            self.doc_id = document_id_for_source(self.source_id, self.content_hash)

    def to_manifest_dict(
        self,
        chunk_count: int = 0,
        *,
        chunking_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        """生成不包含全文的文档清单记录。"""

        methods = sorted({method for block in self.blocks for method in [block.extraction_method]})
        encodings = sorted(
            {
                block.encoding
                for block in self.blocks
                if block.encoding is not None
            }
        )
        pages = sorted(
            {
                block.page_number
                for block in self.blocks
                if block.page_number is not None
            }
        )
        warnings = list(self.warnings)
        for block in self.blocks:
            warnings.extend(block.warnings)
        return {
            "doc_id": self.doc_id,
            "source_id": self.source_id,
            "source_path": self.source_path,
            "file_type": self.file_type,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "block_count": len(self.blocks),
            "chunk_count": chunk_count,
            "pages": pages,
            "extraction_methods": methods,
            "encodings": encodings,
            "warnings": sorted(set(warnings)),
            "ingestion_version": INGESTION_VERSION,
            "ingestion_fingerprint": self.ingestion_fingerprint,
            "chunking_version": CHUNKING_VERSION,
            "chunking_fingerprint": (
                chunking_fingerprint
                if chunking_fingerprint is not None
                else self.chunking_fingerprint
            ),
        }


class IngestionError(RuntimeError):
    """可预期的单文件导入错误。"""


@dataclass(frozen=True)
class FailedDocument:
    """批量导入时记录失败文件，但不阻塞其他文件。"""

    source_path: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_path": self.source_path,
            "error_type": self.error_type,
            "message": self.message,
        }
