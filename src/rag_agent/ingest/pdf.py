"""PDF 页级文本提取和可选 OCR 降级。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import (
    DocumentRecord,
    IngestionError,
    TextBlock,
    config_fingerprint,
    document_id_for_source,
    sha256_bytes,
    source_id_for_path,
)
from .ocr import ocr_page_text


@dataclass(frozen=True)
class PdfOptions:
    """PDF 解析保护参数。"""

    use_ocr: bool = False
    ocr_language: str = "chi_sim+eng"
    ocr_dpi: int = 300
    min_native_chars: int = 40
    max_pages: int = 500

    def fingerprint(self) -> str:
        """返回会影响解析结果的 PDF/OCR 配置指纹。"""

        return config_fingerprint(
            "ingestion",
            {
                "file_type": "pdf",
                "use_ocr": self.use_ocr,
                "ocr_language": self.ocr_language,
                "ocr_dpi": self.ocr_dpi,
                "min_native_chars": self.min_native_chars,
                "max_pages": self.max_pages,
                "extractor": "pymupdf-page-blocks-v1",
            },
        )


def load_pdf_document(
    path: Path,
    *,
    doc_id: str | None = None,
    options: PdfOptions | None = None,
    source_id: str | None = None,
    content_hash: str | None = None,
    ingestion_fingerprint: str | None = None,
) -> DocumentRecord:
    """按页读取 PDF；只有文本质量不足的页才尝试 OCR。"""

    path = path.expanduser().resolve(strict=False)
    options = options or PdfOptions()
    source_id = source_id or source_id_for_path(path)
    content_hash = content_hash or sha256_bytes(path.read_bytes())
    if not doc_id or doc_id == content_hash:
        doc_id = document_id_for_source(source_id, content_hash)
    ingestion_fingerprint = ingestion_fingerprint or options.fingerprint()
    fitz = _import_pymupdf()
    try:
        pdf = fitz.open(str(path))
    except Exception as error:
        raise IngestionError(f"无法打开 PDF：{error}") from error

    try:
        if getattr(pdf, "needs_pass", False):
            raise IngestionError("PDF 受密码保护，当前版本不支持自动解密")
        if len(pdf) > options.max_pages:
            raise IngestionError(
                f"PDF 页数 {len(pdf)} 超过上限 {options.max_pages}，请拆分文件后再导入"
            )

        blocks: list[TextBlock] = []
        for index in range(len(pdf)):
            page_number = index + 1
            page = pdf[index]
            native_text = _native_page_text(page)
            text = native_text
            method = "pdf_text"
            warnings: list[str] = []

            if _needs_ocr(native_text, options.min_native_chars):
                if options.use_ocr:
                    try:
                        ocr_text = ocr_page_text(
                            page,
                            language=options.ocr_language,
                            dpi=options.ocr_dpi,
                        ).strip()
                        if ocr_text:
                            text = ocr_text
                            method = "pdf_ocr"
                        else:
                            warnings.append("ocr_returned_empty_text")
                    except IngestionError as error:
                        # 保留原生文本（即使很短），并把 OCR 问题记录下来；单页失败
                        # 不应该阻塞同一批次的其他文件。
                        warnings.append(f"ocr_unavailable:{error}")
                else:
                    warnings.append("ocr_required_but_disabled")

            blocks.append(
                TextBlock(
                    doc_id=doc_id,
                    source_path=str(path),
                    file_type="pdf",
                    text=text.strip(),
                    extraction_method=method,
                    page_number=page_number,
                    source_id=source_id,
                    warnings=tuple(warnings),
                )
            )

        return DocumentRecord(
            doc_id=doc_id,
            source_path=str(path),
            file_type="pdf",
            content_hash=content_hash,
            size_bytes=path.stat().st_size,
            blocks=blocks,
            warnings=[],
            source_id=source_id,
            ingestion_fingerprint=ingestion_fingerprint,
        )
    finally:
        pdf.close()


def _import_pymupdf() -> Any:
    """兼容新旧包名：PyMuPDF 新版可用 pymupdf，旧版常用 fitz。"""

    try:
        import pymupdf

        return pymupdf
    except ModuleNotFoundError:
        try:
            import fitz

            return fitz
        except ModuleNotFoundError as error:
            raise IngestionError(
                '解析 PDF 需要 PyMuPDF。请执行 `python -m pip install ".[pdf]"`'
            ) from error


def _native_page_text(page: Any) -> str:
    """按版面顺序拼接页面文本块。"""

    try:
        blocks = page.get_text("blocks", sort=True)
    except TypeError:
        blocks = page.get_text("blocks")
    parts = [str(block[4]).strip() for block in blocks if len(block) >= 5]
    return "\n".join(part for part in parts if part)


def _needs_ocr(text: str, min_native_chars: int) -> bool:
    """用简单、可解释的质量指标判断页面是否可能是扫描页。"""

    stripped = text.strip()
    if len(stripped) < min_native_chars:
        return True
    if not stripped:
        return True
    replacement_ratio = stripped.count("\ufffd") / len(stripped)
    if replacement_ratio > 0.02:
        return True
    printable = sum(
        1 for char in stripped if char.isprintable() or char.isspace()
    )
    return printable / len(stripped) < 0.85
