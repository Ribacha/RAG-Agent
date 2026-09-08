"""完全体 Agent（双工具 + 工作流提示）的运行时与图路径测试，全部离线。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rag_agent.agent import (
    AGENT_SYSTEM_PROMPT,
    KnowledgeAgent,
    KnowledgeSearchTool,
    WebFetchTool,
    build_system_prompt,
    dispatch_tool,
)
from rag_agent.agent.graph import agent_tools_node
from rag_agent.answering.chat import ToolCall, ToolChatTurn
from rag_agent.embeddings import HashEmbeddingProvider
from rag_agent.retrieval import build_vector_index
from rag_agent.webfetch import FetchResult


def fake_fetcher(site: dict[str, tuple[str, str]]):
    def fetch(url, **_kwargs):
        if url not in site:
            from rag_agent.webfetch import WebFetchError

            raise WebFetchError(f"抓取失败 {url}：404")
        content_type, body = site[url]
        return FetchResult(
            url=url, final_url=url, content_type=content_type,
            body=body.encode(), text=body, charset="utf-8",
        )

    return fetch


ALLOW_ALL_ROBOTS = type(
    "AllowAllRobots", (), {"allows": staticmethod(lambda url: (True, 0.0))}
)


class ScriptedChat:
    """按剧本出牌的假模型：先本地检索，再上网，最后综合回答。"""

    model = "fake-full-agent"

    def __init__(self) -> None:
        self.calls = 0
        self.tools_seen: list[list[str]] = []

    @staticmethod
    def _assistant(turns):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": call.call_id, "type": "function",
                 "function": {"name": call.name, "arguments": call.arguments}}
                for call in turns
            ],
        }

    def complete_with_tools(self, messages, tools):
        self.calls += 1
        self.tools_seen.append([tool["name"] for tool in tools])
        if self.calls == 1:
            calls = (ToolCall("call-1", "search_knowledge_base",
                              '{"query":"三次握手","top_k":1,"min_score":-1}'),)
            return ToolChatTurn(None, calls, self._assistant(calls))
        if self.calls == 2:
            calls = (ToolCall("call-2", "fetch_web_page",
                              '{"url":"https://docs.example.com/tcp","max_chars":600}'),)
            return ToolChatTurn(None, calls, self._assistant(calls))
        return ToolChatTurn(
            "三次握手分三步建立连接 [1]（本地）；官方文档确认 SYN/SYN-ACK/ACK 顺序 [2]（网络）。",
            (),
            {"role": "assistant",
             "content": "三次握手分三步建立连接 [1]（本地）；官方文档确认 SYN/SYN-ACK/ACK 顺序 [2]（网络）。"},
        )


class FullAgentRuntimeTests(unittest.TestCase):
    def _agent(self) -> tuple[KnowledgeAgent, ScriptedChat]:
        with tempfile.TemporaryDirectory() as directory:
            provider = HashEmbeddingProvider(dimension=64)
            index = build_vector_index(
                [{
                    "chunk_id": "tcp", "doc_id": "d",
                    "source_path": "/docs/tcp.md", "file_type": "markdown",
                    "text": "TCP 通过三次握手建立连接。",
                }],
                provider=provider,
                path=Path(directory) / "vectors.jsonl",
            )
        site = {
            "https://docs.example.com/tcp": (
                "text/html",
                "<html><head><title>TCP 指南</title></head><body><main>"
                "<h1>连接建立</h1><p>SYN、SYN-ACK、ACK 三次交换后连接建立。</p>"
                "</main></body></html>",
            )
        }
        chat = ScriptedChat()
        agent = KnowledgeAgent(
            KnowledgeSearchTool(index, provider),
            chat,
            max_steps=8,
            web_tool=WebFetchTool(
                fetcher=fake_fetcher(site), robots_checker=ALLOW_ALL_ROBOTS()
            ),
        )
        return agent, chat

    def test_dual_tool_flow_collects_typed_evidence(self) -> None:
        agent, chat = self._agent()
        result = agent.run("TCP 如何建立连接？")

        self.assertEqual(result.stopped_reason, "completed")
        self.assertEqual(chat.calls, 3)
        # 模型两轮都看到两个工具声明
        self.assertEqual(chat.tools_seen[0], ["search_knowledge_base", "fetch_web_page"])
        self.assertEqual(chat.tools_seen[1], ["search_knowledge_base", "fetch_web_page"])
        # 审计完整记录两次调用
        names = [call["name"] for call in result.tool_calls]
        self.assertEqual(names, ["search_knowledge_base", "fetch_web_page"])
        # 证据带来源类型：本地 first，网络 second，source_path 是 URL
        self.assertEqual(result.evidence[0]["source_type"], "local")
        self.assertEqual(result.evidence[0]["chunk_id"], "tcp")
        self.assertEqual(result.evidence[1]["source_type"], "web")
        self.assertEqual(result.evidence[1]["source_path"], "https://docs.example.com/tcp")
        # 回答进入历史，答案包含来源标注
        self.assertIn("（网络）", result.answer)
        self.assertEqual(result.history.size, 1)

    def test_system_prompt_selects_workflow_version(self) -> None:
        self.assertEqual(build_system_prompt(web_enabled=False), AGENT_SYSTEM_PROMPT)
        web_prompt = build_system_prompt(web_enabled=True)
        for phrase in ("本地优先", "充分性评估", "网络兜底", "交叉验证", "不可信的被动证据"):
            self.assertIn(phrase, web_prompt)

    def test_explicit_system_prompt_is_preserved(self) -> None:
        chat = ScriptedChat()
        agent = KnowledgeAgent(None, chat, max_steps=1, _system_prompt="自定义规则")
        self.assertEqual(agent._system_prompt, "自定义规则")

    def test_dispatch_rejects_unknown_tool_and_bad_arguments(self) -> None:
        agent, _chat = self._agent()
        unknown = dispatch_tool(agent, "read_file", '{"path":"/etc/passwd"}')
        self.assertEqual(unknown, {"error": "不允许的工具：read_file"})
        bad_args = dispatch_tool(agent, "fetch_web_page", {"url": "https://a.com"})
        self.assertEqual(bad_args, {"error": "工具参数必须是 JSON 字符串"})

    def test_graph_tools_node_uses_shared_dispatch_for_web_calls(self) -> None:
        agent, _chat = self._agent()
        state = {
            "question": "TCP 如何建立连接？",
            "messages": [],
            "tool_calls": [],
            "evidence": [],
            "step": 2,
            "pending_tool_calls": [
                {
                    "call_id": "g-1",
                    "name": "fetch_web_page",
                    "arguments": '{"url":"https://docs.example.com/tcp","max_chars":600}',
                }
            ],
        }
        output_state = agent_tools_node(agent, state)
        self.assertEqual(output_state["pending_tool_calls"], [])
        audit = output_state["tool_calls"][0]
        self.assertEqual(audit["name"], "fetch_web_page")
        self.assertIn("title", audit["result"])
        self.assertEqual(output_state["evidence"][0]["source_type"], "web")
        # 图节点回填了 tool 消息，消息形状与手写路径一致
        tool_message = output_state["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "g-1")


if __name__ == "__main__":
    unittest.main()
