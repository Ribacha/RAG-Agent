"""HTML 清洗与结构化抽取：把网页变成带标题层级的 TextBlock。

"整理"这一步的职责是去噪和保结构：

- 去噪：删掉 script/style/导航/页脚等与正文无关的元素，优先在
  ``<main>``/``<article>`` 容器里取正文；
- 保结构：``h1``-``h6`` 生成标题栈，段落继承 ``heading_path``。这样网页
  的分块和 Markdown 导入一样携带章节元数据，检索结果的引用可以显示
  "来源 URL，章节：xxx"。

抽取使用 BeautifulSoup（可选依赖 ``web`` extra）；与 PDF 依赖的处理方式
一致，未安装时给出明确的安装提示而不是裸 traceback。
"""

from __future__ import annotations

import re
import urllib.parse

from ..models import (
    DocumentRecord,
    TextBlock,
    config_fingerprint,
    document_id_for_source,
    sha256_bytes,
)
from .fetch import FetchResult, normalize_url, source_id_for_url


EXTRACTOR_NAME = "bs4-main-v1"

# 这些元素与正文无关（脚本、样式、整站导航），直接移除。
_NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
    "form",
    "button",
    "select",
    "input",
    "textarea",
    "nav",
    "header",
    "footer",
    "aside",
    "link",
    "meta",
)

# 有 role/aria 标记的导航类区块同样视为噪声。
_NOISE_ROLES = ("navigation", "banner", "contentinfo", "complementary")

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# 产生独立文本块的元素。
_BLOCK_TAGS = {"p", "li", "blockquote", "pre", "dd", "dt", "figcaption", "td", "th"}

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")


class ExtractError(RuntimeError):
    """HTML 抽取阶段的可预期错误（如缺少依赖、空页面）。"""


def _require_bs4():
    try:
        from bs4 import BeautifulSoup  # noqa: PLC0415 - 延迟导入保持启动轻量
    except ModuleNotFoundError as error:
        raise ExtractError(
            "网页抽取需要 BeautifulSoup：请先执行 python -m pip install -e \".[web]\""
        ) from error
    return BeautifulSoup


def _collapse(text: str) -> str:
    """合并空白但不引入语义变化；CJK 文本不受影响。"""

    return _WHITESPACE.sub(" ", text.replace("\n", " ")).strip()


def _clean_soup(soup) -> None:
    """就地移除噪声元素（脚本、样式、导航类区块）。"""

    for node in soup.find_all(_NOISE_TAGS):
        node.decompose()
    # Sphinx 系文档（Python/很多官方文档）在标题里放 "¶" 锚点链接，
    # 属于纯标记噪声。
    for node in soup.find_all(class_="headerlink"):
        node.decompose()
    for node in soup.find_all(attrs={"role": lambda value: value in _NOISE_ROLES}):
        node.decompose()
    for node in soup.find_all(attrs={"aria-hidden": "true"}):
        node.decompose()


def _pick_container(soup):
    """优先使用 main/article 容器，否则退回 body。

    退回 body 时在文档上记录 warning，让 manifest 可以统计"该页面没有
    语义容器，正文可能是整页噪声"。
    """

    for name in ("main", "article"):
        found = soup.find(name)
        if found is not None:
            return found, []
    body = soup.body or soup
    return body, ["no-semantic-container"]


# 有结构意义的元素：容器里若找不到它们，就整体视为一个文本块。
_STRUCTURE_TAGS = frozenset(_HEADING_TAGS) | _BLOCK_TAGS | {"table", "ul", "ol"}


def _table_text(table) -> str:
    """把表格压成逐行文本：单元格用 “ | ” 连接，行间换行。"""

    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [_collapse(cell.get_text(" ", strip=True)) for cell in row.find_all(["td", "th"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _iter_list_items(list_node):
    """深度优先展开列表项；嵌套列表在父项文本之后输出，不重复计数。"""

    for item in list_node.find_all("li", recursive=False):
        nested = item.find_all(["ul", "ol"], recursive=False)
        for node in nested:
            node.extract()  # 从父项文本里摘出来，避免内容重复
        text = _collapse(item.get_text(" ", strip=True))
        if text:
            yield text
        for node in nested:
            yield from _iter_list_items(node)


def extract_blocks(html: str) -> tuple[list[tuple[str, tuple[str, ...]]], list[str], str]:
    """把 HTML 抽成 (正文文本, 标题路径) 列表、诊断 warnings 和页面标题。

    纯函数、离线、确定性：同样的 HTML 永远得到同样的 block 序列。
    """

    BeautifulSoup = _require_bs4()
    soup = BeautifulSoup(html, "html.parser")
    _clean_soup(soup)
    container, warnings = _pick_container(soup)

    title = ""
    if soup.title is not None:
        title = _collapse(soup.title.get_text(" ", strip=True))
    if not title:
        first_h1 = container.find("h1")
        if first_h1 is not None:
            title = _collapse(first_h1.get_text(" ", strip=True))
        warnings.append("title-missing")
    heading_stack: list[str] = [title] if title else []

    blocks: list[tuple[str, tuple[str, ...]]] = []

    def emit(text: str, stack: tuple[str, ...]) -> None:
        if text:
            blocks.append((text, stack))

    def walk(node, stack: list[str]) -> None:
        for child in node.children:
            name = getattr(child, "name", None)
            if name is None or name in _NOISE_TAGS:
                continue
            if name in _HEADING_TAGS:
                level = _HEADING_TAGS[name]
                text = _collapse(child.get_text(" ", strip=True)).rstrip("¶").rstrip()
                if text:
                    # 栈的第 0 位是页面标题（固定根），所以弹出位置是 level
                    # 而不是 level-1：h1 挂在标题之下，h2 挂在 h1 之下。
                    del stack[level:]
                    stack.append(text)
                continue
            if name == "table":
                emit(_table_text(child), tuple(stack))
                continue
            if name in ("ul", "ol"):
                for item in _iter_list_items(child):
                    emit(item, tuple(stack))
                continue
            if name in _BLOCK_TAGS:
                emit(_collapse(child.get_text(" ", strip=True)), tuple(stack))
                continue
            # 普通容器（div/span/section…）：没有结构子元素时整体作为文本块，
            # 否则深入。很多站点（尤其 JS 渲染的 SPA）把正文放在 span/div 里
            # 而不是 <p> 里，这一步保证这类内容不被丢掉。
            if child.find(list(_STRUCTURE_TAGS)) is None:
                emit(_collapse(child.get_text(" ", strip=True)), tuple(stack))
                continue
            walk(child, stack)

    walk(container, heading_stack)
    return blocks, warnings, title


def build_web_document(page: FetchResult) -> DocumentRecord:
    """把一次抓取结果整理成 RAG 数据契约里的 DocumentRecord。

    - ``source_path`` 用规范化 URL（引用显示的就是它）；
    - ``content_hash`` 基于原始字节，与文件导入语义对齐，重爬时用来判断
      内容是否变化；
    - 文本块在"规范化全文"中的字符偏移被如实记录，分块器据此生成
      chunk 级引用区间。
    """

    if page.file_type == "txt":
        body_text = page.text
        raw_blocks: list[tuple[str, tuple[str, ...]]] = [
            (_collapse(paragraph), ())
            for paragraph in re.split(r"\n\s*\n", body_text)
            if _collapse(paragraph)
        ]
        warnings = ["plain-text-page"]
        title = ""
    else:
        raw_blocks, warnings, title = extract_blocks(page.text)

    source_path = page.url
    source_id = source_id_for_url(source_path)
    content_hash = sha256_bytes(page.body)
    doc_id = document_id_for_source(source_id, content_hash)
    ingestion_fingerprint = config_fingerprint(
        "ingestion",
        {
            "file_type": "html",
            "extractor": EXTRACTOR_NAME,
            "collapse_whitespace": True,
        },
    )

    # 组装规范化全文并记录每个 block 的偏移区间：块之间用空行分隔，
    # 与 Markdown 规范化策略一致。
    blocks: list[TextBlock] = []
    normalized_parts: list[str] = []
    offset = 0
    for text, heading_path in raw_blocks:
        start = offset
        normalized_parts.append(text)
        offset += len(text) + 2  # 块间两个分隔符长度
        blocks.append(
            TextBlock(
                doc_id=doc_id,
                source_path=source_path,
                file_type="html",
                text=text,
                extraction_method="html",
                heading_path=heading_path,
                source_id=source_id,
                source_char_start=start,
                source_char_end=start + len(text),
                normalized_char_start=start,
                normalized_char_end=start + len(text),
                encoding=page.charset,
            )
        )
    if not blocks:
        warnings.append("empty-page")

    return DocumentRecord(
        doc_id=doc_id,
        source_path=source_path,
        file_type="html",
        content_hash=content_hash,
        size_bytes=len(page.body),
        blocks=blocks,
        warnings=sorted(set(warnings)),
        source_id=source_id,
        ingestion_fingerprint=ingestion_fingerprint,
    )


def extract_links(html: str, base_url: str) -> list[str]:
    """抽取页面里的站内链接（绝对化、去 fragment、保序去重）。

    供爬虫决定下一步抓什么；只返回 http/https 链接。发现链接前先做与
    正文抽取相同的去噪：导航/页脚里的工具链接（索引页、bug 页等）不该
    进入抓取范围。
    """

    BeautifulSoup = _require_bs4()
    soup = BeautifulSoup(html, "html.parser")
    _clean_soup(soup)

    seen: set[str] = set()
    ordered: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith(("mailto:", "javascript:", "tel:", "#")):
            continue
        if anchor.find_parent(_NOISE_TAGS) is not None:
            continue
        try:
            absolute = normalize_url(urllib.parse.urljoin(base_url, href))
        except Exception:
            continue
        if absolute not in seen:
            seen.add(absolute)
            ordered.append(absolute)
    return ordered
