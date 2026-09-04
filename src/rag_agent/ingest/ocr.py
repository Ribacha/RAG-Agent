"""OCR 后端适配器。

当前默认调用 PyMuPDF 的 get_textpage_ocr。它仍需要系统安装 Tesseract 和对应
语言包；把调用封装在这里，后续可以替换为 PaddleOCR、Apple Vision 或云 OCR。
"""

from __future__ import annotations

import shutil
from typing import Any

from ..models import IngestionError


def ocr_page_text(page: Any, *, language: str, dpi: int) -> str:
    """对单页执行 OCR，缺少系统后端时抛出可读错误。"""

    if shutil.which("tesseract") is None:
        raise IngestionError(
            "未找到 tesseract。请先执行 `brew install tesseract tesseract-lang`，"
            "再确认 `tesseract --list-langs` 包含 chi_sim 和 eng。"
        )

    try:
        text_page = page.get_textpage_ocr(
            language=language,
            dpi=dpi,
            full=True,
        )
        return page.get_text("text", textpage=text_page)
    except Exception as error:  # PyMuPDF/Tesseract 的异常类型随版本变化
        raise IngestionError(f"Tesseract OCR 失败：{error}") from error
