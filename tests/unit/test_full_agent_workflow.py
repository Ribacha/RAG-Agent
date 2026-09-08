"""完全体 Agent 工作流的四路径离线剧本测试。

对应 docs/Agent工作流工程文档.md 第 2 节的编排图：
1. 本地足够 → 不上网；2. 本地不足 → 模型选 URL 多跳；
3. 索引为空 → 先检索确认后直接网络；4. 双双不足 → 明确拒答。
外加一条 LangGraph 执行路径的对等性验证。全部离线。
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rag_agent.agent import (
    KnowledgeAgent,
    KnowledgeSearchTool,
    WebFetchTool,
    run_knowledge_graph,
)
from rag_agent.answering.chat import ToolCall, ToolChatTurn
from rag_agent.embeddings import HashEmbeddingProvider
from rag_agent.retrieval import build_vector_index
from rag_agent.webfetch import FetchResult


ALLOW_ALL_ROBOTS = type(
    "AllowAllRobots", (), {"allows": staticmethod(lambda url: (True, 0.0))}
)


def fake_fetcher(site: dict[str, tuple[str, str]]):
    def fetch(url, **_kwargs):
        if url not in site:
            from rag_agent.webfetch import WebFetchError

            raise WebFetchError(f"抓取失败 {url}：HTTP Error 404")
        content_type, body = site[url]
        return FetchResult(
            url=url, final_url=url, content_type=content_type,
            body=body.encode(), text=body, charset="utf-8",
        )

    return fetch


def forbidden_fetcher(url, **_kwargs):
    raise AssertionError("这条工作流路径不应访问网络")


class ScriptedChat:
    """按剧本出牌：每步要么是一组工具调用，要么是字符串最终回答。"""

    model = "fake-workflow"

    def __init__(self, steps: list) -> None:
        self.steps = list(steps)
        self.cursor = 0

    def complete_with_tools(self, messages, tools):
        if self.cursor >= len(self.steps):
            raise AssertionError("剧本耗尽：模型调用了超出预期的轮数")
        step = self.steps[self.cursor]
        self.cursor += 1
        if isinstance(step, str):
            return ToolChatTurn(step, (), {"role": "assistant", "content": step})
        assistant = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": call.call_id, "type": "function",
                 "function": {"name": call.name, "arguments": call.arguments}}
                for call in step
            ],
        }
        return ToolChatTurn(None, tuple(step), assistant)


def _index(chunks: list[dict]):
    directory = tempfile.TemporaryDirectory()
    provider = HashEmbeddingProvider(dimension=64)
    index = build_vector_index(
        chunks, provider=provider, path=Path(directory.name) / "vectors.jsonl"
    )
    return index, provider, directory


class WorkflowPathTests(unittest.TestCase):
    def _agent(self, *, chunks, steps, site=None, web_fetcher=None):
        index, provider, directory = _index(chunks)
        self.addCleanup(directory.cleanup)
        fetcher = web_fetcher or (
            forbidden_fetcher if site is None else fake_fetcher(site)
        )
        chat = ScriptedChat(steps)
        agent = KnowledgeAgent(
            KnowledgeSearchTool(index, provider),
            chat,
            max_steps=8,
            web_tool=WebFetchTool(fetcher=fetcher, robots_checker=ALLOW_ALL_ROBOTS()),
        )
        return agent, chat

    def test_path1_local_sufficient_never_touches_web(self) -> None:
        agent, _chat = self._agent(
            chunks=[{
                "chunk_id": "tcp", "doc_id": "d",
                "source_path": "/docs/tcp.md", "file_type": "markdown",
                "text": "TCP 通过三次握手建立连接：SYN、SYN-ACK、ACK。",
            }],
            steps=[
                (ToolCall("c1", "search_knowledge_base",
                          '{"query":"三次握手","top_k":3,"min_score":-1}'),),
                "三次握手分三步：SYN、SYN-ACK、ACK [1]（本地）。",
            ],
        )
        result = agent.run("TCP 三次握手的过程？")

        self.assertEqual(result.stopped_reason, "completed")
        self.assertEqual(
            [call["name"] for call in result.tool_calls],
            ["search_knowledge_base"],
        )
        self.assertTrue(result.evidence)
        self.assertTrue(all(item["source_type"] == "local" for item in result.evidence))
        # forbidden_fetcher 若被调用会直接 AssertionError，测试通过即证明没上网

    def test_path2_local_insufficient_multi_hop_web(self) -> None:
        site = {
            "https://docs.example.com/guide": (
                "text/html",
                '<html><head><title>指南</title></head><body><main>'
                "<h1>概览</h1><p>本页只有概览；细节见进阶章节。</p>"
                '<a href="/guide-advanced">进阶</a>'
                "</main></body></html>",
            ),
            "https://docs.example.com/guide-advanced": (
                "text/html",
                "<html><head><title>进阶</title></head><body><main>"
                "<h1>进阶细节</h1><p>关键结论：参数 X 默认值为 42。</p>"
                "</main></body></html>",
            ),
        }
        agent, _chat = self._agent(
            chunks=[{
                "chunk_id": "irrelevant", "doc_id": "d",
                "source_path": "/docs/other.md", "file_type": "markdown",
                "text": "完全无关的内容：天气与穿搭。",
            }],
            steps=[
                (ToolCall("c1", "search_knowledge_base",
                          '{"query":"参数 X 默认值","top_k":3,"min_score":0.6}'),),
                (ToolCall("c2", "fetch_web_page",
                          '{"url":"https://docs.example.com/guide","max_chars":600}'),),
                (ToolCall("c3", "fetch_web_page",
                          '{"url":"https://docs.example.com/guide-advanced","max_chars":600}'),),
                "概览页没有答案，进阶章节给出结论：参数 X 默认值为 42 [1]（网络）。",
            ],
            site=site,
        )
        result = agent.run("参数 X 的默认值是多少？")

        self.assertEqual(result.stopped_reason, "completed")
        names = [call["name"] for call in result.tool_calls]
        self.assertEqual(
            names, ["search_knowledge_base", "fetch_web_page", "fetch_web_page"]
        )
        # 两跳都在预算内，且第二跳的 URL 正是第一跳返回的链接
        self.assertEqual(agent.web_tool.fetches_used, 2)
        web_urls = [
            call["result"]["url"]
            for call in result.tool_calls
            if call["name"] == "fetch_web_page"
        ]
        self.assertEqual(
            web_urls,
            ["https://docs.example.com/guide", "https://docs.example.com/guide-advanced"],
        )
        self.assertIn(
            "https://docs.example.com/guide-advanced",
            result.tool_calls[1]["result"]["links"],
        )
        self.assertTrue(all(item["source_type"] == "web" for item in result.evidence))

    def test_path3_empty_index_confirms_then_goes_web(self) -> None:
        site = {
            "https://docs.example.com/faq": (
                "text/html",
                "<html><head><title>FAQ</title></head><body><main>"
                "<p>问题的官方答案是 7 天。</p></main></body></html>",
            )
        }
        agent, _chat = self._agent(
            chunks=[],  # 空索引：检索必然零命中
            steps=[
                (ToolCall("c1", "search_knowledge_base", '{"query":"答案"}'),),
                (ToolCall("c2", "fetch_web_page",
                          '{"url":"https://docs.example.com/faq","max_chars":600}'),),
                "知识库为空，官方 FAQ 给出答案：7 天 [1]（网络）。",
            ],
            site=site,
        )
        result = agent.run("这个问题的答案是什么？")

        self.assertEqual(result.stopped_reason, "completed")
        # 模型先检索确认知识库为空（count=0），再上网
        self.assertEqual(result.tool_calls[0]["name"], "search_knowledge_base")
        self.assertEqual(result.tool_calls[0]["result"]["count"], 0)
        self.assertEqual(result.tool_calls[1]["name"], "fetch_web_page")
        self.assertEqual(result.evidence[0]["source_type"], "web")

    def test_path4_both_fail_refuses_clearly(self) -> None:
        agent, _chat = self._agent(
            chunks=[],
            steps=[
                (ToolCall("c1", "search_knowledge_base", '{"query":"冷门问题"}'),),
                (ToolCall("c2", "fetch_web_page",
                          '{"url":"https://docs.example.com/missing","max_chars":600}'),),
                "本地知识库与网络都没有找到足够依据，无法回答这个问题。",
            ],
            site={},  # 空 site：任何抓取都 404
        )
        result = agent.run("一个哪儿都没有答案的冷门问题")

        self.assertEqual(result.stopped_reason, "completed")
        # 抓取失败以可审计错误进入记录，而不是中断运行
        self.assertIn("404", result.tool_calls[1]["result"]["error"])
        self.assertIn("没有找到足够依据", result.answer)
        # 模型明确拒答是一次"完成"，进入历史
        self.assertEqual(result.history.size, 1)

    def test_graph_path_matches_runtime_for_local_only_flow(self) -> None:
        agent, _chat = self._agent(
            chunks=[{
                "chunk_id": "tcp", "doc_id": "d",
                "source_path": "/docs/tcp.md", "file_type": "markdown",
                "text": "TCP 通过三次握手建立连接。",
            }],
            steps=[
                (ToolCall("c1", "search_knowledge_base",
                          '{"query":"三次握手","top_k":3,"min_score":-1}'),),
                "LangGraph 路径回答 [1]（本地）。",
            ],
        )
        result = run_knowledge_graph(agent, "TCP 三次握手？")
        self.assertEqual(result.stopped_reason, "completed")
        self.assertEqual([c["name"] for c in result.tool_calls], ["search_knowledge_base"])
        self.assertTrue(all(item["source_type"] == "local" for item in result.evidence))


if __name__ == "__main__":
    unittest.main()
