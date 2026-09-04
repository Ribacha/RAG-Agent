"""可重复、可解释的字符级分块器。

第一阶段先用字符数控制块大小，避免把 tokenizer 或某一家模型绑定进导入层。
后续接入 Embedding 后，可以在不改变 Chunk 数据契约的前提下增加 token-aware
策略。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import (
    CHUNKING_VERSION,
    Chunk,
    DocumentRecord,
    TextBlock,
    config_fingerprint,
    sha256_text,
    source_id_for_path,
)


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = 1200
    overlap_chars: int = 120

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars 必须大于 0")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars 不能小于 0")
        if self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars 必须小于 max_chars")

    def fingerprint(self) -> str:
        """返回会影响分块结果的配置指纹。"""

        return config_fingerprint(
            "chunking",
            {
                "version": CHUNKING_VERSION,
                "strategy": "character-natural-boundary-v1",
                "max_chars": self.max_chars,
                "overlap_chars": self.overlap_chars,
            },
        )


def split_text_with_offsets(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[tuple[str, int, int]]:
    """将文本切成块，并返回每块相对于输入文本的半开区间。"""

    if max_chars <= 0:
        raise ValueError("max_chars 必须大于 0")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars 必须满足 0 <= overlap_chars < max_chars")

    # 去掉首尾空白时同步记录偏移，避免 chunk 的来源区间完全失真。
    leading = len(text) - len(text.lstrip())
    source = text.strip()
    if not source:
        return []

    pieces: list[tuple[str, int, int]] = []
    start = 0
    source_length = len(source)
    while start < source_length:
        hard_end = min(start + max_chars, source_length)
        end = hard_end
        if hard_end < source_length:
            # 在块后半段寻找自然边界；中文常用标点和换行都可以作为边界。
            boundary_start = start + max(1, max_chars // 2)
            boundary = _last_boundary(source, boundary_start, hard_end)
            if boundary is not None:
                end = boundary

        raw_piece = source[start:end]
        piece = raw_piece.strip()
        if piece:
            left_trim = len(raw_piece) - len(raw_piece.lstrip())
            right_trim = len(raw_piece) - len(raw_piece.rstrip())
            piece_start = leading + start + left_trim
            piece_end = leading + end - right_trim
            pieces.append((piece, piece_start, piece_end))

        if end >= source_length:
            break

        # 从上一块末尾回退 overlap，确保相邻块保留少量上下文。
        next_start = max(end - overlap_chars, start + 1)
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return pieces


def _last_boundary(text: str, start: int, end: int) -> int | None:
    """返回区间内最后一个适合断开的边界（边界位置之后开始下一块）。"""

    candidates: list[int] = []
    for marker in ("\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "):
        position = text.rfind(marker, start, end)
        if position >= 0:
            candidates.append(position + len(marker))
    return max(candidates) if candidates else None


def chunk_document(
    document: DocumentRecord,
    *,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """将一个文档的 TextBlock 转换为带来源元数据的 Chunk。"""

    config = config or ChunkConfig()
    chunking_fingerprint = config.fingerprint()
    # The CLI writes the manifest after chunking, so retain the exact strategy
    # on the mutable document record for backwards-compatible callers.
    document.chunking_fingerprint = chunking_fingerprint
    source_id = document.source_id or source_id_for_path(document.source_path)
    chunks: list[Chunk] = []
    for block in document.blocks:
        if not block.text.strip():
            continue
        pieces = split_text_with_offsets(
            block.text,
            max_chars=config.max_chars,
            overlap_chars=config.overlap_chars,
        )
        for piece, local_start, local_end in pieces:
            chunk_index = len(chunks)
            content_hash = sha256_text(piece)
            # Include source identity and both processing fingerprints.  A rerun
            # with the same inputs is stable, while a moved file or changed
            # extraction/chunking configuration cannot collide with old chunks.
            chunk_id = sha256_text(
                "|".join(
                    (
                        "chunk:v2",
                        source_id,
                        document.doc_id,
                        document.content_hash,
                        str(chunk_index),
                        content_hash,
                        document.ingestion_fingerprint,
                        chunking_fingerprint,
                    )
                )
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    source_path=document.source_path,
                    file_type=document.file_type,
                    text=piece,
                    chunk_index=chunk_index,
                    content_hash=content_hash,
                    page_start=block.page_number,
                    page_end=block.page_number,
                    heading_path=block.heading_path,
                    source_id=source_id,
                    ingestion_fingerprint=document.ingestion_fingerprint,
                    chunking_fingerprint=chunking_fingerprint,
                    char_start=(
                        (
                            block.normalized_char_start + local_start
                            if block.normalized_char_start is not None
                            else block.source_char_start + local_start
                        )
                        if (
                            block.normalized_char_start is not None
                            or block.source_char_start is not None
                        )
                        else None
                    ),
                    char_end=(
                        (
                            block.normalized_char_start + local_end
                            if block.normalized_char_start is not None
                            else block.source_char_start + local_end
                        )
                        if (
                            block.normalized_char_start is not None
                            or block.source_char_start is not None
                        )
                        else None
                    ),
                    extraction_methods=(block.extraction_method,),
                    warnings=block.warnings,
                )
            )
    return chunks
