"""网页抓取与整理。

把一个网站的文字内容变成现有 RAG 链路的输入。设计约束与 ``ingest/`` 一致：

1. 复用数据契约：网页清洗后直接产出 ``DocumentRecord``/``TextBlock``，
   source_path 用 URL、file_type 用 ``html``，分块、引用元数据和增量向量
   复用机制原样生效；
2. 抓取是确定性的受控流水线（礼貌爬虫），不把爬取决策交给 LLM——爬取要
   快、便宜、可离线测试，LLM 只在问答层使用；
3. 所有网络操作都可注入（fetcher/sleep），单元测试完全离线。
"""

from .fetch import FetchResult, WebFetchError, fetch_url, normalize_url, source_id_for_url
from .crawler import (
    CrawlConfig,
    CrawlFailure,
    CrawlResult,
    CrawledPage,
    crawl_site,
)
from .extract import extract_links, build_web_document
from .ingest_url import merge_web_documents
from .render import RenderFetcher, create_render_fetcher

__all__ = [
    "FetchResult",
    "WebFetchError",
    "fetch_url",
    "normalize_url",
    "source_id_for_url",
    "CrawlConfig",
    "CrawlFailure",
    "CrawlResult",
    "CrawledPage",
    "crawl_site",
    "extract_links",
    "build_web_document",
    "merge_web_documents",
    "RenderFetcher",
    "create_render_fetcher",
]
