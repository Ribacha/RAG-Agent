"""Markdown 和 TXT 的文本读取器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
from typing import Iterable

from ..models import (
    DocumentRecord,
    IngestionError,
    TextBlock,
    config_fingerprint,
    document_id_for_source,
    sha256_bytes,
    source_id_for_path,
)


@dataclass(frozen=True)
class ReadTextResult:
    text: str
    encoding: str
    warnings: tuple[str, ...] = ()


def read_text(path: Path) -> ReadTextResult:
    """读取文本文件，并报告实际使用的编码。

    偏移量在后续阶段基于这里返回的规范化 Unicode 字符串计算。先处理 BOM，
    再尝试 UTF-8 和 charset-normalizer，最后用 GB18030 兜底，避免常见中文
    文件在没有检测器时悄悄变成替换字符。
    """

    raw = path.read_bytes()

    # 必须在 NUL 检查之前识别 UTF-16/32；ASCII UTF-16 文件通常含大量 NUL，
    # 但它们并不是二进制垃圾。
    bom_encodings = (
        (b"\xff\xfe\x00\x00", "utf-32"),
        (b"\x00\x00\xfe\xff", "utf-32"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
    )
    for bom, encoding in bom_encodings:
        if raw.startswith(bom):
            try:
                return ReadTextResult(
                    _normalize_newlines(raw.decode(encoding)),
                    encoding,
                    (),
                )
            except UnicodeDecodeError as error:
                raise IngestionError(
                    f"文本文件的 {encoding} BOM 与内容不匹配：{path.name}"
                ) from error

    # BOM-less UTF-16 ASCII has a strong alternating-NUL pattern and is also
    # technically valid UTF-8.  Handle that pattern before the UTF-8 attempt;
    # ordinary UTF-8 (including Chinese text) does not have it.
    if _has_utf16_nul_pattern(raw, minimum_ratio=0.25):
        utf16_candidate = _decode_bomless_utf16(raw)
        if utf16_candidate is not None:
            return ReadTextResult(
                _normalize_newlines(utf16_candidate),
                "utf-16-heuristic",
                ("encoding_detected_by_utf16_heuristic",),
            )

    # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的 UTF-8。
    try:
        text = raw.decode("utf-8-sig")
        if raw and raw.count(b"\x00") > max(1, len(raw) // 100):
            raise IngestionError(f"疑似二进制文件，拒绝按文本读取：{path.name}")
        return ReadTextResult(_normalize_newlines(text), "utf-8-sig")
    except UnicodeDecodeError:
        pass

    # After UTF-8 has been ruled out, use a lower threshold so short CJK
    # documents containing only one or two newline NULs are still recognized.
    if _has_utf16_nul_pattern(raw, minimum_ratio=0.05):
        utf16_candidate = _decode_bomless_utf16(raw)
        if utf16_candidate is not None:
            return ReadTextResult(
                _normalize_newlines(utf16_candidate),
                "utf-16-heuristic",
                ("encoding_detected_by_utf16_heuristic",),
            )

    if raw and raw.count(b"\x00") > max(1, len(raw) // 100):
        raise IngestionError(f"疑似二进制文件，拒绝按文本读取：{path.name}")

    # charset-normalizer 是正式依赖；保留缺失时的兜底，便于源码目录在安装前
    # 进行最基本的诊断。
    try:
        from charset_normalizer import from_bytes
    except ModuleNotFoundError:
        from_bytes = None

    if from_bytes is not None:
        # `.best()` can choose cp949/UTF-16 with equal confidence for very
        # short Chinese samples.  Inspect all candidates and prefer the known
        # Chinese/UTF-16 candidate with the strongest CJK ratio instead.
        matches = list(from_bytes(raw))
        preferred: list[tuple[float, int, str, str]] = []
        priorities = {
            "gb18030": 0,
            "gb2312": 1,
            "utf_16_le": 2,
            "utf_16_be": 3,
        }
        for match in matches:
            encoding = (match.encoding or "").lower().replace("-", "_")
            if encoding not in priorities:
                continue
            try:
                text = str(match)
            except (UnicodeDecodeError, LookupError):
                continue
            if not text:
                continue
            cjk_ratio = sum("\u3400" <= char <= "\u9fff" for char in text) / len(text)
            preferred.append((cjk_ratio, -priorities[encoding], encoding, text))
        if preferred:
            cjk_ratio, _, encoding, text = max(preferred)
            if cjk_ratio >= 0.5:
                return ReadTextResult(
                    _normalize_newlines(text),
                    encoding,
                    ("encoding_detected_by_charset_normalizer",),
                )

    # charset-normalizer 对极短的中文样本可能没有结论；GB18030 是常见的
    # 中文本地文件编码，严格解码成功时优先保留可读文本。
    try:
        text = raw.decode("gb18030")
        return ReadTextResult(
            _normalize_newlines(text),
            "gb18030",
            ("encoding_fallback_gb18030",),
        )
    except UnicodeDecodeError:
        pass

    # 最后的兜底会保留替换字符，并明确写 warning；不能静默生成看似正常的乱码。
    text = raw.decode("utf-8", errors="replace")
    return ReadTextResult(
        _normalize_newlines(text),
        "utf-8-with-replacement",
        ("encoding_fallback_with_replacement",),
    )


def _decode_bomless_utf16(raw: bytes) -> str | None:
    """Return a plausible BOM-less UTF-16 decoding, if one is evident."""

    if len(raw) < 2 or len(raw) % 2:
        return None
    # Without a BOM, a reliable signal is the NUL pattern produced by mostly
    # ASCII UTF-16 text.  GB18030/Big5/ordinary binary data commonly has no
    # such pattern; do not let a coincidental even byte length turn Chinese
    # legacy text into mojibake.
    half_length = len(raw) / 2
    even_nuls = raw[0::2].count(0) / half_length
    odd_nuls = raw[1::2].count(0) / half_length
    endian_signal = max(even_nuls, odd_nuls)
    if endian_signal < 0.05:
        return None
    if odd_nuls > even_nuls + 0.05:
        preferred_order = ("utf-16-le", "utf-16-be")
    elif even_nuls > odd_nuls + 0.05:
        preferred_order = ("utf-16-be", "utf-16-le")
    else:
        preferred_order = ("utf-16-le", "utf-16-be")
    candidates: list[tuple[float, int, str]] = []
    for order, encoding in enumerate(preferred_order):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if not text:
            continue
        printable = sum(
            1 for char in text if char.isprintable() or char.isspace()
        ) / len(text)
        controls = sum(
            1
            for char in text
            if unicodedata.category(char) == "Cc" and not char.isspace()
        ) / len(text)
        cjk = sum("\u3400" <= char <= "\u9fff" for char in text) / len(text)
        # CJK content and the absence of control characters are useful signals;
        # printable ASCII UTF-16 is distinguished by its NUL-byte distribution.
        line_breaks = sum(char in "\r\n" for char in text) / len(text)
        score = (
            printable
            - controls * 2.0
            + cjk * 0.35
            + line_breaks * 1.5
            + (1.0 - order) * endian_signal
        )
        candidates.append((score, order, text))
    if not candidates:
        return None
    score, _, text = max(candidates, key=lambda item: (item[0], -item[1]))
    if score < 0.75:
        return None
    # Avoid treating ordinary binary data as UTF-16 merely because a decode
    # happened to succeed.
    if any(unicodedata.category(char) == "Cs" for char in text):
        return None
    return text


def _has_utf16_nul_pattern(raw: bytes, *, minimum_ratio: float = 0.25) -> bool:
    """Whether alternating NUL bytes strongly suggest BOM-less UTF-16."""

    if len(raw) < 2 or len(raw) % 2:
        return False
    half = len(raw) // 2
    even_ratio = raw[0::2].count(0) / half
    odd_ratio = raw[1::2].count(0) / half
    # A single newline in a short CJK document can lower the ratio below 0.25;
    # 5% still distinguishes UTF-16's alternating NULs from ordinary GB18030.
    return max(even_ratio, odd_ratio) >= minimum_ratio


def _normalize_newlines(text: str) -> str:
    """统一换行符，但保留空行和正文内容。"""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_text_document(
    path: Path,
    *,
    doc_id: str | None = None,
    file_type: str = "txt",
    source_id: str | None = None,
    content_hash: str | None = None,
    ingestion_fingerprint: str | None = None,
) -> DocumentRecord:
    """读取 TXT 文件，或作为 Markdown 的底层读取实现。

    TXT 先保留为一个完整 block；真正的段落/长度切分由独立 chunker 完成，避免
    文本读取和分块策略互相耦合。
    """

    path = path.expanduser().resolve(strict=False)
    result = read_text(path)
    source_id = source_id or source_id_for_path(path)
    content_hash = content_hash or sha256_bytes(path.read_bytes())
    if not doc_id or doc_id == content_hash:
        doc_id = document_id_for_source(source_id, content_hash)
    ingestion_fingerprint = ingestion_fingerprint or config_fingerprint(
        "ingestion",
        {
            "file_type": file_type,
            "normalization": "crlf_to_lf",
            "encoding": result.encoding,
        },
    )
    block = TextBlock(
        doc_id=doc_id,
        source_path=str(path),
        file_type=file_type,
        text=result.text,
        extraction_method="txt" if file_type == "txt" else "markdown",
        source_char_start=0,
        source_char_end=len(result.text),
        source_id=source_id,
        normalized_char_start=0,
        normalized_char_end=len(result.text),
        encoding=result.encoding,
        warnings=result.warnings,
    )
    return DocumentRecord(
        doc_id=doc_id,
        source_path=str(path),
        file_type=file_type,
        content_hash=content_hash,
        size_bytes=path.stat().st_size,
        blocks=[block],
        warnings=[],
        source_id=source_id,
        ingestion_fingerprint=ingestion_fingerprint,
    )


def iter_lines_with_offsets(text: str) -> Iterable[tuple[str, int, int]]:
    """逐行返回正文（不含换行符）及其在原文中的区间。"""

    position = 0
    for line in text.splitlines(keepends=True):
        end = position + len(line)
        yield line.rstrip("\r\n"), position, end
        position = end
    if position < len(text):
        yield text[position:], position, len(text)


def heading_match(line: str) -> tuple[int, str] | None:
    """识别 CommonMark 风格的 ATX 标题，返回层级和标题文本。"""

    match = re.match(r"^\s{0,3}(#{1,6})[ \t]+(.+?)\s*$", line)
    if match is None:
        return None
    title = re.sub(r"[ \t]+#+[ \t]*$", "", match.group(2)).strip()
    return len(match.group(1)), title
