"""保留标题层级的 Markdown 加载器。"""

from __future__ import annotations

from pathlib import Path

from ..models import (
    DocumentRecord,
    TextBlock,
    config_fingerprint,
    document_id_for_source,
    sha256_bytes,
    source_id_for_path,
)
from .text import heading_match, iter_lines_with_offsets, read_text


def load_markdown_document(
    path: Path,
    *,
    doc_id: str | None = None,
    source_id: str | None = None,
    content_hash: str | None = None,
    ingestion_fingerprint: str | None = None,
) -> DocumentRecord:
    """读取 Markdown，并为每个段落记录当前 heading_path。"""

    path = path.expanduser().resolve(strict=False)
    result = read_text(path)
    source_id = source_id or source_id_for_path(path)
    content_hash = content_hash or sha256_bytes(path.read_bytes())
    if not doc_id or doc_id == content_hash:
        doc_id = document_id_for_source(source_id, content_hash)
    ingestion_fingerprint = ingestion_fingerprint or config_fingerprint(
        "ingestion",
        {
            "file_type": "markdown",
            "normalization": "crlf_to_lf",
            "encoding": result.encoding,
            "heading_parser": "atx-v1",
        },
    )
    blocks: list[TextBlock] = []
    heading_stack: list[str] = []
    lines: list[str] = []
    block_start: int | None = None
    block_end: int | None = None
    in_fence = False

    def flush() -> None:
        nonlocal lines, block_start, block_end
        if block_start is None or block_end is None:
            lines = []
            return
        # The lines are contiguous slices of the normalized source.  Slicing
        # that source directly preserves an honest character interval; joining
        # lines would change CRLF/whitespace positions and make citations drift.
        raw_text = result.text[block_start:block_end]
        left_trim = len(raw_text) - len(raw_text.lstrip())
        right_trim = len(raw_text) - len(raw_text.rstrip())
        text = raw_text.strip()
        if text:
            # 当前 block 的 heading_path 在收集期间保持不变；复制一份避免后续
            # 标题更新修改已完成 block 的元数据。
            blocks.append(
                TextBlock(
                    doc_id=doc_id,
                    source_path=str(path),
                    file_type="markdown",
                    text=text,
                    extraction_method="markdown",
                    heading_path=tuple(heading_stack),
                    source_char_start=block_start,
                    source_char_end=block_end,
                    source_id=source_id,
                    normalized_char_start=block_start + left_trim,
                    normalized_char_end=block_end - right_trim,
                    encoding=result.encoding,
                    warnings=result.warnings,
                )
            )
        lines = []
        block_start = None
        block_end = None

    for line, start, end in iter_lines_with_offsets(result.text):
        stripped = line.strip()
        is_fence = stripped.startswith("```") or stripped.startswith("~~~")
        if is_fence:
            if block_start is None:
                block_start = start
            lines.append(line)
            block_end = end
            in_fence = not in_fence
            continue

        heading = None if in_fence else heading_match(line)
        if heading is not None:
            # 标题前的正文属于旧标题上下文。
            flush()
            level, title = heading
            del heading_stack[level - 1 :]
            heading_stack.append(title)
            block_start = start
            block_end = end
            lines = [line]
            continue

        if not stripped and not in_fence:
            flush()
            continue

        if block_start is None:
            block_start = start
        lines.append(line)
        block_end = end

    flush()

    return DocumentRecord(
        doc_id=doc_id,
        source_path=str(path),
        file_type="markdown",
        content_hash=content_hash,
        size_bytes=path.stat().st_size,
        blocks=blocks,
        warnings=[],
        source_id=source_id,
        ingestion_fingerprint=ingestion_fingerprint,
    )
