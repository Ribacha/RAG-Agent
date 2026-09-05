from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_agent.cli import main
from rag_agent.webfetch import FetchResult, crawl_site, extract_links
from rag_agent.webfetch.crawler import CrawlConfig
from rag_agent.webfetch.ingest_url import merge_web_documents

try:
    import bs4  # noqa: F401

    BS4_AVAILABLE = True
except ModuleNotFoundError:
    BS4_AVAILABLE = False


SITE_HTML = """
<html>
<head><title>测试站点</title><style>body { color: red; }</style></head>
<body>
<nav><a href="/nav-link">导航</a></nav>
<main>
  <h1>计算机网络讲义</h1>
  <p>TCP 通过三次握手建立连接。</p>
  <h2>流量控制</h2>
  <p>接收方通过滑动窗口控制发送速率。</p>
  <ul><li>停止等待</li><li>滑动窗口</li></ul>
  <a href="/chapter2.html">第二章</a>
  <a href="https://other.example.com/outside">外站链接</a>
</main>
<footer>版权所有</footer>
<script>console.log("noise");</script>
</body>
</html>
"""


def fake_result(url: str, body: str, content_type: str = "text/html") -> FetchResult:
    raw = body.encode("utf-8")
    return FetchResult(
        url=url,
        final_url=url,
        content_type=content_type,
        body=raw,
        text=body,
        charset="utf-8",
    )


def fake_site_fetcher(site: dict[str, tuple[str, str]]):
    """构造一个离线假 fetcher：URL -> (content_type, body)。"""

    def fetcher(url, **_kwargs):
        if url not in site:
            from rag_agent.webfetch import WebFetchError

            raise WebFetchError(f"抓取失败 {url}：HTTPError: HTTP Error 404")
        content_type, body = site[url]
        return fake_result(url, body, content_type)

    return fetcher


@unittest.skipUnless(BS4_AVAILABLE, "需要 beautifulsoup4")
class ExtractTests(unittest.TestCase):
    def test_extract_blocks_removes_noise_and_keeps_headings(self) -> None:
        from rag_agent.webfetch.extract import extract_blocks

        blocks, warnings, title = extract_blocks(SITE_HTML)
        texts = [text for text, _ in blocks]

        self.assertEqual(title, "测试站点")
        self.assertIn("TCP 通过三次握手建立连接。", texts)
        self.assertIn("接收方通过滑动窗口控制发送速率。", texts)
        # 噪声不应出现在正文里
        self.assertFalse(any("console.log" in text for text in texts))
        self.assertFalse(any("版权所有" in text for text in texts))
        self.assertFalse(any("导航" == text for text in texts))
        # 列表项被展开成独立 block
        self.assertIn("停止等待", texts)
        # 标题层级继承：h2 下的段落 heading_path 以页面标题和章节为栈
        path = next(path for text, path in blocks if text.startswith("接收方"))
        self.assertEqual(path, ("测试站点", "计算机网络讲义", "流量控制"))
        self.assertEqual(warnings, [])

    def test_extract_links_absolute_and_deduplicated(self) -> None:
        html = """
        <a href="/a.html">A</a>
        <a href="/a.html">A 重复</a>
        <a href="b.html#frag">B</a>
        <a href="mailto:a@b.c">Mail</a>
        <a href="https://ext.example.com/c">C</a>
        <nav><a href="/nav-tool.html">导航里的工具链接</a></nav>
        <footer><a href="/footer-tool.html">页脚链接</a></footer>
        """
        links = extract_links(html, "https://test.local/index.html")
        self.assertEqual(
            links,
            [
                "https://test.local/a.html",
                "https://test.local/b.html",
                "https://ext.example.com/c",
            ],
        )

    def test_heading_strips_anchor_marks(self) -> None:
        from rag_agent.webfetch.extract import extract_blocks

        html = "<html><head><title>T</title></head><body><main>" \
               "<h2>流量控制<span class=\"headerlink\">¶</span></h2>" \
               "<p>内容。</p></main></body></html>"
        blocks, _warnings, _title = extract_blocks(html)
        headings = {path[-1] for _, path in blocks if path}
        self.assertIn("流量控制", headings)
        self.assertNotIn("流量控制 ¶", headings)

    def test_build_web_document_uses_url_as_source(self) -> None:
        from rag_agent.webfetch.extract import build_web_document

        record = build_web_document(fake_result("https://test.local/", SITE_HTML))
        self.assertEqual(record.source_path, "https://test.local/")
        self.assertEqual(record.file_type, "html")
        self.assertTrue(record.blocks)
        self.assertTrue(all(block.source_path == "https://test.local/" for block in record.blocks))
        # 同一页面重复整理结果完全一致（确定性）
        again = build_web_document(fake_result("https://test.local/", SITE_HTML))
        self.assertEqual(record.doc_id, again.doc_id)
        self.assertEqual(record.ingestion_fingerprint, again.ingestion_fingerprint)


class CrawlerTests(unittest.TestCase):
    def _site(self) -> dict[str, tuple[str, str]]:
        robots = "User-agent: *\nDisallow: /private\n"
        index = (
            "<html><head><title>Index</title></head><body>"
            "<a href='/page1.html'>P1</a>"
            "<a href='/private.html'>私有</a>"
            "<a href='https://else.example.com/x'>外站</a>"
            "</body></html>"
        )
        return {
            "https://test.local/robots.txt": ("text/plain", robots),
            "https://test.local/": ("text/html", index),
            "https://test.local/page1.html": ("text/html", "<html><body><p>第一章内容</p></body></html>"),
            "https://test.local/private.html": ("text/html", "<html><body><p>私有内容</p></body></html>"),
        }

    def test_crawl_respects_scope_robots_and_depth(self) -> None:
        site = self._site()
        slept: list[float] = []
        result = crawl_site(
            "https://test.local/",
            CrawlConfig(max_pages=10, max_depth=1, delay_seconds=0.5),
            fetcher=fake_site_fetcher(site),
            sleep=slept.append,
        )

        page_urls = [page.url for page in result.pages]
        self.assertEqual(page_urls, ["https://test.local/", "https://test.local/page1.html"])
        # private.html 被 robots 拦截，外站被同域规则过滤
        failure_urls = {failure.url for failure in result.failures}
        self.assertIn("https://test.local/private.html", failure_urls)
        self.assertNotIn("https://else.example.com/x", failure_urls)
        self.assertTrue(all("else.example.com" not in url for url in page_urls))
        # 限速生效：第二个请求前睡 0.5 秒
        self.assertEqual(slept, [0.5])

    def test_crawl_stops_at_max_pages(self) -> None:
        site = self._site()
        site["https://test.local/"] = (
            "text/html",
            "<html><body>"
            + "".join(f"<a href='/p{i}.html'>{i}</a>" for i in range(5))
            + "</body></html>",
        )
        for i in range(5):
            site[f"https://test.local/p{i}.html"] = ("text/html", f"<html><body><p>{i}</p></body></html>")

        result = crawl_site(
            "https://test.local/",
            CrawlConfig(max_pages=3, max_depth=1, delay_seconds=0),
            fetcher=fake_site_fetcher(site),
            sleep=lambda _seconds: None,
        )
        self.assertEqual(len(result.pages), 3)
        self.assertTrue(all(failure.error_type != "robots-disallowed" for failure in result.failures))

    def test_crawl_records_network_failures_without_stopping(self) -> None:
        site = self._site()
        site["https://test.local/"] = (
            "text/html",
            "<html><body><a href='/missing.html'>404</a><a href='/page1.html'>P1</a></body></html>",
        )
        result = crawl_site(
            "https://test.local/",
            CrawlConfig(max_pages=5, max_depth=1, delay_seconds=0),
            fetcher=fake_site_fetcher(site),
            sleep=lambda _seconds: None,
        )
        self.assertEqual(len(result.pages), 2)  # 入口 + page1
        self.assertIn("https://test.local/missing.html", {f.url for f in result.failures})


class MergeTests(unittest.TestCase):
    def test_merge_keeps_uncrawled_documents_and_counts_changes(self) -> None:
        from rag_agent.webfetch.extract import build_web_document

        record = build_web_document(fake_result("https://test.local/", SITE_HTML))
        empty_chunks: list[dict] = []
        empty_manifests: list[dict] = []
        first = merge_web_documents([record], empty_chunks, empty_manifests, chunk_config=None)
        self.assertEqual(first.counts.added, 1)

        # 同一内容再次合并：计为 unchanged，chunk_id 完全相同
        second = merge_web_documents([record], first.chunks, first.manifests, chunk_config=None)
        self.assertEqual(second.counts.unchanged, 1)
        first_ids = {chunk["chunk_id"] for chunk in first.chunks}
        second_ids = {chunk["chunk_id"] for chunk in second.chunks}
        self.assertEqual(first_ids, second_ids)

        # 另一个页面：旧数据应原样保留
        other = build_web_document(fake_result("https://test.local/other.html", "<html><body><p>其他</p></body></html>"))
        third = merge_web_documents([other], first.chunks, first.manifests, chunk_config=None)
        self.assertEqual(third.counts.added, 1)
        kept = {chunk["source_path"] for chunk in third.chunks}
        self.assertIn("https://test.local/", kept)


class IngestUrlCliTests(unittest.TestCase):
    def _run(self, argv: list[str]) -> tuple[int, str]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(argv)
        return exit_code, output.getvalue()

    def test_ingest_url_builds_searchable_index_and_reuses_vectors(self) -> None:
        site = {
            "https://test.local/robots.txt": ("text/plain", "User-agent: *\nAllow: /\n"),
            "https://test.local/": (
                "text/html",
                "<html><head><title>讲义</title></head><body><main>"
                "<h1>网络层</h1><p>IP 协议负责寻址和路由。</p>"
                "</main></body></html>",
            ),
        }
        fetcher = fake_site_fetcher(site)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "ingest-url",
                "https://test.local/",
                "--output", str(root / "chunks.jsonl"),
                "--manifest", str(root / "documents.jsonl"),
                "--failures", str(root / "failures.jsonl"),
                "--index", str(root / "vectors.jsonl"),
                "--delay", "0",
            ]
            with patch("rag_agent.webfetch.crawler.fetch_url", fetcher):
                exit_code, rendered = self._run(argv)
            self.assertEqual(exit_code, 0)
            payload = json.loads(rendered[rendered.index("{"): rendered.rindex("}") + 1])
            self.assertEqual(payload["pages_crawled"], 1)
            self.assertEqual(payload["pages_added"], 1)
            self.assertEqual(payload["vectors_embedded"], payload["chunks_written"])

            chunks = [json.loads(line) for line in (root / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(chunk["source_path"] == "https://test.local/" for chunk in chunks))
            self.assertTrue(all(chunk["file_type"] == "html" for chunk in chunks))

            search_output = io.StringIO()
            with contextlib.redirect_stdout(search_output):
                exit_code = main(
                    [
                        "search",
                        "IP 协议负责寻址和路由",
                        "--index", str(root / "vectors.jsonl"),
                        "--top-k", "3",
                        "--min-score", "-1",
                        "--json",
                    ]
                )
            self.assertEqual(exit_code, 0)
            results = json.loads(search_output.getvalue())
            self.assertTrue(results)
            self.assertEqual(results[0]["source_path"], "https://test.local/")

            # 第二次抓取同样内容：unchanged 且全部向量复用，不重新消耗 embedding
            with patch("rag_agent.webfetch.crawler.fetch_url", fetcher):
                exit_code, rendered = self._run(argv)
            self.assertEqual(exit_code, 0)
            payload = json.loads(rendered[rendered.index("{"): rendered.rindex("}") + 1])
            self.assertEqual(payload["pages_unchanged"], 1)
            self.assertEqual(payload["vectors_embedded"], 0)
            self.assertEqual(payload["vectors_reused"], payload["chunks_written"])

    def test_ingest_url_reports_failures_with_nonzero_exit(self) -> None:
        site = {
            "https://test.local/robots.txt": ("text/plain", "User-agent: *\nDisallow: /\n"),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch(
                "rag_agent.webfetch.crawler.fetch_url", fake_site_fetcher(site)
            ):
                exit_code = main(
                    [
                        "ingest-url",
                        "https://test.local/",
                        "--output", str(root / "chunks.jsonl"),
                        "--manifest", str(root / "documents.jsonl"),
                        "--failures", str(root / "failures.jsonl"),
                        "--index", str(root / "vectors.jsonl"),
                        "--delay", "0",
                    ]
                )
            self.assertEqual(exit_code, 2)
            failures = [
                json.loads(line)
                for line in (root / "failures.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(failures[0]["error_type"], "robots-disallowed")


if __name__ == "__main__":
    unittest.main()
