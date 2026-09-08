"""WebFetchTool 的离线单元测试：全部护栏与输出契约。"""

from __future__ import annotations

import json
import unittest

from rag_agent.agent.web_tool import (
    FETCH_WEB_PAGE_TOOL,
    NullSearchProvider,
    WebFetchTool,
    WebToolError,
    _is_blocked_host,
)
from rag_agent.webfetch import FetchResult


def fake_page(url: str, body: str, content_type: str = "text/html") -> FetchResult:
    return FetchResult(
        url=url,
        final_url=url,
        content_type=content_type,
        body=body.encode("utf-8"),
        text=body,
        charset="utf-8",
    )


def site_fetcher(site: dict[str, tuple[str, str]]):
    """URL -> (content_type, body) 的离线假 fetcher；未收录的 URL 视为 404。"""

    def fetch(url, **_kwargs):
        if url not in site:
            from rag_agent.webfetch import WebFetchError

            raise WebFetchError(f"抓取失败 {url}：HTTPError: 404")
        content_type, body = site[url]
        return fake_page(url, body, content_type)

    return fetch


ALLOW_ALL_ROBOTS = type(
    "AllowAllRobots",
    (),
    {"allows": staticmethod(lambda url: (True, 0.0))},
)


class GuardTests(unittest.TestCase):
    """不触网的参数与地址校验。"""

    def test_tool_schema_shape(self) -> None:
        self.assertEqual(FETCH_WEB_PAGE_TOOL["name"], "fetch_web_page")
        self.assertEqual(FETCH_WEB_PAGE_TOOL["parameters"]["required"], ["url"])
        self.assertTrue(FETCH_WEB_PAGE_TOOL["strict"])

    def test_blocked_hosts(self) -> None:
        for host in (
            "localhost",
            "127.0.0.1",
            "10.1.2.3",
            "192.168.1.1",
            "172.16.0.9",
            "172.31.255.255",
            "169.254.169.254",
            "0.0.0.0",
            "::1",
            "fe80::1",
            "fd12::1",
        ):
            self.assertTrue(_is_blocked_host(host), host)
        for host in ("example.com", "docs.python.org", "8.8.8.8", "1.1.1.1"):
            self.assertFalse(_is_blocked_host(host), host)

    def test_invoke_rejects_internal_addresses_before_any_fetch(self) -> None:
        calls: list[str] = []

        def fetch(url, **_kwargs):
            calls.append(url)
            raise AssertionError("不应发起抓取")

        tool = WebFetchTool(fetcher=fetch, robots_checker=ALLOW_ALL_ROBOTS())
        for url in (
            "http://localhost:8000/admin",
            "http://127.0.0.1/x",
            "http://10.0.0.5/internal",
            "https://192.168.1.1/router",
        ):
            with self.assertRaises(WebToolError) as raised:
                tool.invoke(url)
            self.assertIn("SSRF", str(raised.exception))
        self.assertEqual(calls, [])

    def test_bad_arguments(self) -> None:
        tool = WebFetchTool(fetcher=None, respect_robots=False, robots_checker=ALLOW_ALL_ROBOTS())
        for bad in (
            "not-json",
            "[1,2]",
            json.dumps({"url": "https://a.com", "extra": 1}),
            json.dumps({"url": 123}),
            json.dumps({"url": "https://a.com", "max_chars": "big"}),
            json.dumps({"url": "https://a.com", "max_chars": 10}),
        ):
            with self.assertRaises(WebToolError):
                tool.invoke_json(bad)
        with self.assertRaises(WebToolError):
            tool.invoke("ftp://example.com/file")


class FetchTests(unittest.TestCase):
    def _tool(self, site: dict[str, tuple[str, str]], **kwargs) -> WebFetchTool:
        return WebFetchTool(
            fetcher=site_fetcher(site),
            robots_checker=ALLOW_ALL_ROBOTS(),
            **kwargs,
        )

    def test_render_page_extracts_title_text_and_links(self) -> None:
        tool = self._tool({})
        page = fake_page(
            "https://docs.example.com/guide",
            "<html><head><title>指南</title></head><body><main>"
            "<h1>快速开始</h1><p>第一步安装依赖。</p>"
            '<a href="/advanced">进阶</a><a href="https://other.example.com/x">外站</a>'
            "</main></body></html>",
        )
        output = tool.render_page(page, max_chars=500)
        self.assertEqual(output["title"], "指南")
        self.assertIn("第一步安装依赖。", output["text"])
        self.assertIn("https://docs.example.com/advanced", output["links"])
        self.assertFalse(output["truncated"])
        self.assertEqual(output["source_type"], "web")
        # results 与检索工具对齐：运行时证据归集可直接消费
        self.assertEqual(output["results"][0]["source_path"], "https://docs.example.com/guide")
        self.assertEqual(output["results"][0]["source_type"], "web")

    def test_render_page_truncates_long_text(self) -> None:
        tool = self._tool({})
        page = fake_page(
            "https://docs.example.com/long",
            "<html><body><main><p>" + "很长的内容" * 500 + "</p></main></body></html>",
        )
        output = tool.render_page(page, max_chars=600)
        self.assertTrue(output["truncated"])
        self.assertLessEqual(len(output["text"]), 600)

    def test_invoke_success_and_json_contract(self) -> None:
        site = {
            "https://docs.example.com/": (
                "text/html",
                "<html><head><title>首页</title></head><body><main><p>正文。</p></main></body></html>",
            )
        }
        tool = self._tool(site)
        raw = tool.invoke_json(json.dumps({"url": "https://docs.example.com/"}))
        payload = json.loads(raw)
        self.assertEqual(payload["title"], "首页")
        self.assertEqual(tool.fetches_used, 1)

    def test_plain_text_page(self) -> None:
        site = {"https://example.com/notes.txt": ("text/plain", "纯文本笔记\n第二行")}
        tool = self._tool(site)
        output = tool.invoke("https://example.com/notes.txt")
        self.assertIn("纯文本笔记", output["text"])
        self.assertEqual(output["links"], [])
        self.assertIn("plain-text-page", output["warnings"])

    def test_fetch_budget_capped_per_run(self) -> None:
        site = {
            f"https://docs.example.com/p{i}": ("text/html", f"<html><body><main><p>第{i}页</p></main></body></html>")
            for i in range(5)
        }
        tool = self._tool(site, max_fetches=2)
        tool.invoke("https://docs.example.com/p0")
        tool.invoke("https://docs.example.com/p1")
        with self.assertRaises(WebToolError) as raised:
            tool.invoke("https://docs.example.com/p2")
        self.assertIn("抓取上限", str(raised.exception))

    def test_robots_disallowed_blocks_fetch(self) -> None:
        site = {
            "https://private.example.com/robots.txt": (
                "text/plain",
                "User-agent: *\nDisallow: /\n",
            ),
        }
        tool = WebFetchTool(fetcher=site_fetcher(site))  # 真实 _RobotsCache 路径
        with self.assertRaises(WebToolError) as raised:
            tool.invoke("https://private.example.com/secret")
        self.assertIn("robots", str(raised.exception))

    def test_fetch_failure_becomes_auditable_error(self) -> None:
        tool = self._tool({})  # 空 site：任何 URL 都是 404
        with self.assertRaises(WebToolError) as raised:
            tool.invoke("https://docs.example.com/missing")
        self.assertIn("404", str(raised.exception))


class SearchProviderTests(unittest.TestCase):
    def test_null_provider_returns_empty(self) -> None:
        provider = NullSearchProvider()
        self.assertEqual(provider.name, "null")
        self.assertEqual(provider.search("任何问题"), [])


if __name__ == "__main__":
    unittest.main()
