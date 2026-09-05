"""可选的 JS 渲染抓取：用无头浏览器处理动态渲染的网站。

静态抓取（fetch.py）拿到的动态站点 HTML 是一个空壳——正文由浏览器里的
JavaScript 现场生成。本模块提供一个符合爬虫 fetcher 约定的替代实现：

1. 每个请求先走静态路径：robots.txt、text/plain 等非 HTML 资源不需要
   浏览器，直接返回；404/超限等错误也在这一层暴露；
2. HTML 页面交给 Playwright（Chromium 无头）渲染，等 JS 执行完再取
   最终 DOM，之后走完全相同的清洗/分块/索引链路。

Playwright 与浏览器二进制都是重量级可选依赖（``[web-js]`` extra +
``playwright install chromium``），缺失时给出明确安装指引；整个模块
不进入默认依赖，静态抓取路径完全不受影响。
"""

from __future__ import annotations

from .fetch import (
    DEFAULT_USER_AGENT,
    FetchResult,
    WebFetchError,
    fetch_url,
    normalize_url,
)


_HTML_TYPES = ("text/html", "application/xhtml+xml")


class RenderFetcher:
    """符合 ``crawl_site(fetcher=...)`` 约定的渲染抓取器。

    ``page_renderer`` 是"URL -> (最终 URL, 渲染后 HTML)"的函数，默认用
    Playwright 实现；测试注入假实现即可离线覆盖逻辑。
    """

    def __init__(
        self,
        *,
        timeout_ms: int = 20000,
        wait_until: str = "networkidle",
        user_agent: str = DEFAULT_USER_AGENT,
        page_renderer=None,
        static_fetcher=None,
    ) -> None:
        self._timeout_ms = timeout_ms
        self._wait_until = wait_until
        self._user_agent = user_agent
        self._page_renderer = page_renderer
        self._static_fetcher = static_fetcher or fetch_url
        # 显式传入 page_renderer 时完全不触碰 Playwright（测试离线运行）；
        # 否则立刻启动真实引擎，依赖缺失在创建时就会暴露，而不是爬到一半。
        self._engine = None if page_renderer is not None else _build_playwright_renderer(
            timeout_ms=timeout_ms,
            wait_until=wait_until,
            user_agent=user_agent,
        )

    def __call__(
        self,
        url: str,
        *,
        timeout: float = 15.0,
        max_bytes: int = 5 * 1024 * 1024,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> FetchResult:
        # 先走静态路径：robots.txt、纯文本资源直接返回，404/超限也在这里
        # 暴露，浏览器只用于真正需要渲染的 HTML 页面。
        static = self._static_fetcher(
            url, timeout=timeout, max_bytes=max_bytes, user_agent=user_agent
        )
        if static.content_type not in _HTML_TYPES:
            return static

        renderer = self._page_renderer or self._engine
        try:
            final_url, html = renderer(
                static.url,
                timeout_ms=self._timeout_ms,
                user_agent=self._user_agent,
            )
        except WebFetchError:
            raise
        except Exception as error:
            raise WebFetchError(
                f"渲染失败 {static.url}：{type(error).__name__}: {error}"
            ) from error

        body = html.encode("utf-8")
        if len(body) > max_bytes:
            raise WebFetchError(f"页面超过大小上限 {max_bytes} 字节：{static.url}")
        return FetchResult(
            url=static.url,
            final_url=normalize_url(final_url),
            content_type="text/html",
            body=body,
            text=html,
            charset="utf-8",
        )

    def close(self) -> None:
        """释放浏览器资源；CLI 在抓取结束后必须调用。"""

        engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()


def _build_playwright_renderer(*, timeout_ms: int, wait_until: str, user_agent: str):
    """启动一次 Chromium 并返回带 close 的渲染引擎；缺失依赖时给出安装指引。"""

    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as error:
        raise WebFetchError(
            "动态渲染需要 Playwright：请执行 "
            'python -m pip install -e ".[web-js]" && '
            "python -m playwright install chromium"
        ) from error

    try:
        driver = sync_playwright().start()
        browser = driver.chromium.launch(headless=True)
        context = browser.new_context(user_agent=user_agent)
    except Exception as error:
        raise WebFetchError(
            f"无法启动 Chromium 浏览器：{error}。"
            "请确认已执行 python -m playwright install chromium"
        ) from error

    class _PlaywrightEngine:
        """URL -> (最终 URL, 渲染后 HTML)，资源只在 close 时释放。"""

        def __call__(
            self,
            url: str,
            *,
            timeout_ms: int = timeout_ms,
            user_agent: str = user_agent,
        ) -> tuple[str, str]:
            page = context.new_page()
            try:
                try:
                    page.goto(url, timeout=timeout_ms, wait_until=wait_until)
                except Exception:
                    # networkidle 在常驻连接（统计脚本等）的站点上会超时，
                    # 退回 load 事件再取 DOM。
                    page.goto(url, timeout=timeout_ms, wait_until="load")
                page.wait_for_timeout(300)  # 留一点最后一帧渲染时间
                return page.url, page.content()
            finally:
                page.close()

        def close(self) -> None:
            try:
                context.close()
                browser.close()
            finally:
                driver.stop()

    return _PlaywrightEngine()


def create_render_fetcher(
    *,
    timeout_ms: int = 20000,
    wait_until: str = "networkidle",
    user_agent: str = DEFAULT_USER_AGENT,
) -> RenderFetcher:
    """创建使用真实 Chromium 的渲染抓取器；依赖缺失立刻抛 WebFetchError。"""

    return RenderFetcher(
        timeout_ms=timeout_ms,
        wait_until=wait_until,
        user_agent=user_agent,
    )
