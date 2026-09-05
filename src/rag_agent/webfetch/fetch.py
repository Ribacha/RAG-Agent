"""HTTP 抓取：单页下载、URL 规范化和来源身份。

与文件导入的区别在于"文件路径"换成了 URL，因此来源身份必须基于规范化
URL 计算，而不是 ``source_id_for_path``（后者会经过 Path.resolve，会把
URL 当成本地路径处理而得到不稳定结果）。
"""

from __future__ import annotations

from dataclasses import dataclass
import urllib.parse
import urllib.request

from ..models import sha256_bytes


DEFAULT_USER_AGENT = "rag-agent-crawler/0.3 (+https://github.com/Ribacha/RAG-Agent)"

# 只接受网页类内容，避免把二进制资源（图片、压缩包）当文本导入。
SUPPORTED_CONTENT_TYPES = {
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/plain": "txt",
}


class WebFetchError(RuntimeError):
    """可预期的抓取错误（网络、协议、内容类型、大小上限）。"""


def normalize_url(url: str) -> str:
    """规范化 URL：只允许 http/https，去掉 fragment。

    fragment（#后的部分）不改变服务器返回的内容，保留它会让同一页面
    产生多个来源身份；大小写只规范化 scheme 和 host 部分（路径可能
    区分大小写，保持原样）。
    """

    url = (url or "").strip()
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise WebFetchError(f"只支持 http/https 链接：{url!r}")
    if not parsed.netloc:
        raise WebFetchError(f"URL 缺少主机名：{url!r}")
    return urllib.parse.urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.query,
            "",
        )
    )


def source_id_for_url(url: str) -> str:
    """按规范化 URL 生成来源身份，与 ``models.source_id_for_path`` 同构。"""

    return sha256_bytes(("source:v1:" + normalize_url(url)).encode("utf-8"))


@dataclass(frozen=True)
class FetchResult:
    """一次成功下载的结果。

    ``body`` 保留原始字节用于内容哈希（与文件导入的 ``sha256_bytes`` 语义
    对齐），``text`` 是按响应头声明编码解码后的文本。
    """

    url: str
    final_url: str
    content_type: str
    body: bytes
    text: str
    charset: str

    @property
    def file_type(self) -> str:
        return SUPPORTED_CONTENT_TYPES.get(self.content_type, "html")


def fetch_url(
    url: str,
    *,
    timeout: float = 15.0,
    max_bytes: int = 5 * 1024 * 1024,
    user_agent: str = DEFAULT_USER_AGENT,
) -> FetchResult:
    """下载单个 URL 并解码为文本；失败抛 ``WebFetchError``。

    读取限制在 ``max_bytes`` 内：超限直接失败而不是静默截断，避免把
    半个页面当完整内容导入索引。
    """

    normalized = normalize_url(url)
    request = urllib.request.Request(
        normalized,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            final_url = response.geturl() or normalized
            body = response.read(max_bytes + 1)
    except WebFetchError:
        raise
    except Exception as error:  # urllib 会抛 HTTPError/URLError/TimeoutError 等
        raise WebFetchError(f"抓取失败 {normalized}：{type(error).__name__}: {error}") from error

    if len(body) > max_bytes:
        raise WebFetchError(
            f"页面超过大小上限 {max_bytes} 字节：{normalized}"
        )
    if content_type and content_type not in SUPPORTED_CONTENT_TYPES:
        raise WebFetchError(f"不支持的 Content-Type：{content_type}（{normalized}）")

    text = body.decode(charset, errors="replace")
    return FetchResult(
        url=normalized,
        final_url=normalize_url(final_url),
        content_type=content_type or "text/html",
        body=body,
        text=text,
        charset=charset,
    )
