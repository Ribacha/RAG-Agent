"""礼貌爬虫：从入口 URL 广度优先抓取同一站点的网页。

为什么是确定性调度而不是 LLM 决策：

- 爬取要面对真实网站，礼貌与边界（robots.txt、同域、页数上限、限速）
  必须被强制执行，不能依赖模型的自觉；
- 确定性调度让抓取结果可复现，失败清单可解释，也完全离线可测。

所有网络与睡眠依赖都通过参数注入，测试用假 fetcher 即可模拟整站。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import urllib.parse
import urllib.robotparser

from .extract import extract_links
from .fetch import FetchResult, WebFetchError, fetch_url, normalize_url


@dataclass(frozen=True)
class CrawlConfig:
    """一次抓取的边界配置；数值默认值偏向保守。"""

    max_pages: int = 10
    max_depth: int = 1
    delay_seconds: float = 1.0
    respect_robots: bool = True
    same_domain: bool = True
    timeout: float = 15.0
    max_page_bytes: int = 5 * 1024 * 1024
    user_agent: str = "rag-agent-crawler/0.3 (+https://github.com/Ribacha/RAG-Agent)"

    def __post_init__(self) -> None:
        if self.max_pages <= 0:
            raise ValueError("max_pages 必须大于 0")
        if self.max_depth < 0:
            raise ValueError("max_depth 不能小于 0")
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds 不能小于 0")


@dataclass(frozen=True)
class CrawledPage:
    """一个成功抓取并保留的页面。"""

    url: str
    depth: int
    result: FetchResult


@dataclass(frozen=True)
class CrawlFailure:
    """一个没能抓取的 URL 与原因；不阻塞其他页面。"""

    url: str
    error_type: str
    message: str


@dataclass(frozen=True)
class CrawlResult:
    """整站抓取的汇总。"""

    pages: tuple[CrawledPage, ...]
    failures: tuple[CrawlFailure, ...]
    queued: int  # 进入过队列的 URL 总数（含重复发现），用于观察边界是否生效

    @property
    def links_found(self) -> int:
        return max(self.queued - len(self.pages) - len(self.failures), 0)


class _RobotsCache:
    """按站点缓存 robots.txt 解析结果。

    robots.txt 拿不到（404/网络错误）时选择放行：单页限制靠 max_pages 和
    同域约束兜底，站点 robots 缺失不应导致整个功能不可用。
    """

    def __init__(self, fetcher, user_agent: str) -> None:
        self._fetcher = fetcher
        self._user_agent = user_agent
        self._cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _parser_for(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        parsed = urllib.parse.urlsplit(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        if robots_url in self._cache:
            return self._cache[robots_url]
        try:
            result = self._fetcher(robots_url)
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(result.text.splitlines())
            self._cache[robots_url] = parser
        except Exception:
            parser = None
            self._cache[robots_url] = None
        return parser

    def allows(self, url: str) -> tuple[bool, float]:
        """返回 (是否允许抓取, 站点要求的 crawl-delay 秒数)。"""

        parser = self._parser_for(url)
        if parser is None:
            return True, 0.0
        allowed = parser.can_fetch(self._user_agent, url)
        delay = parser.crawl_delay(self._user_agent)
        if delay is None:
            delay = parser.crawl_delay("*")
        return allowed, float(delay) if delay else 0.0


def crawl_site(
    start_url: str,
    config: CrawlConfig,
    *,
    fetcher=None,
    sleep=time.sleep,
) -> CrawlResult:
    """从入口 URL 广度优先抓取。

    - ``fetcher``：URL -> FetchResult 的函数，默认真实 HTTP；测试可注入；
    - ``sleep``：请求间隔；测试注入成空函数。
    """

    fetcher = fetcher or fetch_url
    start = normalize_url(start_url)
    start_domain = urllib.parse.urlsplit(start).netloc

    robots = _RobotsCache(fetcher, config.user_agent) if config.respect_robots else None
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(start, 0)]
    pages: list[CrawledPage] = []
    failures: list[CrawlFailure] = []
    queued_count = 1

    while queue and len(pages) < config.max_pages:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        if robots is not None:
            allowed, robots_delay = robots.allows(url)
            if not allowed:
                failures.append(
                    CrawlFailure(url, "robots-disallowed", "robots.txt 不允许抓取该页面")
                )
                continue
            effective_delay = max(config.delay_seconds, robots_delay)
        else:
            effective_delay = config.delay_seconds

        if pages or failures or queued_count > 1:  # 第一个请求之前不休眠
            sleep(effective_delay)
        try:
            result = fetcher(
                url,
                timeout=config.timeout,
                max_bytes=config.max_page_bytes,
                user_agent=config.user_agent,
            )
        except WebFetchError as error:
            failures.append(CrawlFailure(url, type(error).__name__, str(error)))
            continue
        except Exception as error:  # 注入的 fetcher 可能抛任意异常
            failures.append(CrawlFailure(url, type(error).__name__, f"{error}"))
            continue

        pages.append(CrawledPage(url=url, depth=depth, result=result))

        if depth < config.max_depth:
            for link in extract_links(result.text, result.final_url):
                if link in visited:
                    continue
                if config.same_domain and urllib.parse.urlsplit(link).netloc != start_domain:
                    continue
                queued_count += 1
                queue.append((link, depth + 1))

    return CrawlResult(pages=tuple(pages), failures=tuple(failures), queued=queued_count)
