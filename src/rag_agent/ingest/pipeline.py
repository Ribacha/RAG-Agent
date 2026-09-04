"""按扩展名选择加载器，并处理批量导入错误。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..models import (
    DocumentRecord,
    FailedDocument,
    IngestionError,
    document_id_for_source,
    sha256_bytes,
    source_id_for_path,
)
from .markdown import load_markdown_document
from .pdf import PdfOptions, load_pdf_document
from .text import load_text_document


SUPPORTED_SUFFIXES = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "txt",
    ".pdf": "pdf",
}


@dataclass(frozen=True)
class IngestionBatch:
    records: tuple[DocumentRecord, ...]
    failures: tuple[FailedDocument, ...]


def discover_files(input_path: Path, *, allow_empty: bool = False) -> list[Path]:
    """发现一个文件或目录下的受支持文件，返回稳定排序结果。"""

    path = input_path.expanduser().resolve()
    if not path.exists():
        raise IngestionError(f"输入路径不存在：{path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise IngestionError(f"不支持的文件类型：{path.suffix or path.name}")
        return [path]
    if not path.is_dir():
        raise IngestionError(f"输入路径不是文件或目录：{path}")

    files = []
    for candidate in path.rglob("*"):
        # 跳过隐藏目录、符号链接和临时文件，避免越过用户选择的目录边界。
        if candidate.is_symlink() or any(part.startswith(".") for part in candidate.relative_to(path).parts):
            continue
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(candidate.resolve())
    files = sorted(files, key=lambda item: str(item).casefold())
    if not files and not allow_empty:
        raise IngestionError(f"目录中没有找到 PDF、Markdown 或 TXT 文件：{path}")
    return files


def ingest_path(
    input_path: Path,
    *,
    pdf_options: PdfOptions | None = None,
    max_file_bytes: int = 50 * 1024 * 1024,
) -> IngestionBatch:
    """批量导入；单文件失败会进入 failures，不阻塞其他文件。"""

    files = discover_files(input_path)
    records: list[DocumentRecord] = []
    failures: list[FailedDocument] = []
    for path in files:
        try:
            if path.stat().st_size > max_file_bytes:
                raise IngestionError(
                    f"文件大小超过上限 {max_file_bytes} 字节：{path.name}"
                )
            records.append(load_file(path, pdf_options=pdf_options))
        except Exception as error:
            failures.append(
                FailedDocument(
                    source_path=str(path),
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
    return IngestionBatch(tuple(records), tuple(failures))


def load_file(path: Path, *, pdf_options: PdfOptions | None = None) -> DocumentRecord:
    """加载单个受支持文件。"""

    path = path.expanduser().resolve()
    content_hash = sha256_bytes(path.read_bytes())
    source_id = source_id_for_path(path)
    # doc_id identifies one version of one source.  Keeping content_hash as a
    # separate field lets callers detect byte-identical files without merging
    # their source metadata or retrieval citations.
    doc_id = document_id_for_source(source_id, content_hash)
    file_type = SUPPORTED_SUFFIXES[path.suffix.lower()]
    if file_type == "markdown":
        return load_markdown_document(
            path,
            doc_id=doc_id,
            source_id=source_id,
            content_hash=content_hash,
        )
    if file_type == "txt":
        return load_text_document(
            path,
            doc_id=doc_id,
            file_type="txt",
            source_id=source_id,
            content_hash=content_hash,
        )
    if file_type == "pdf":
        return load_pdf_document(
            path,
            doc_id=doc_id,
            options=pdf_options,
            source_id=source_id,
            content_hash=content_hash,
        )
    raise IngestionError(f"不支持的文件类型：{path.suffix}")
