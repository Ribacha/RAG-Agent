from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rag_agent.agent import ConversationHistory, KnowledgeSearchTool
from rag_agent.agent.runtime import KnowledgeAgent
from rag_agent.answering.chat import ToolCall, ToolChatTurn
from rag_agent.embeddings import HashEmbeddingProvider
from rag_agent.retrieval import build_vector_index


class FakeToolChat:
    model = "fake-agent-model"

    def __init__(self) -> None:
        self.calls = 0
        self.messages = []
        self.tools = []

    def complete_with_tools(self, messages, tools):
        self.calls += 1
        self.messages.append(list(messages))
        self.tools.append(list(tools))
        if self.calls == 1:
            return ToolChatTurn(
                content=None,
                tool_calls=(
                    ToolCall(
                        call_id="call-1",
                        name="search_knowledge_base",
                        arguments='{"query":"注意力","top_k":1,"min_score":-1}',
                    ),
                ),
                assistant_message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "search_knowledge_base",
                                "arguments": '{"query":"注意力","top_k":1,"min_score":-1}',
                            },
                        }
                    ],
                },
            )
        return ToolChatTurn(
            content="注意力机制使用 Query、Key 和 Value。[1]",
            tool_calls=(),
            assistant_message={
                "role": "assistant",
                "content": "注意力机制使用 Query、Key 和 Value。[1]",
            },
        )


class AgentRuntimeTests(unittest.TestCase):
    def test_agent_executes_only_declared_search_tool_then_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = HashEmbeddingProvider(dimension=64)
            index = build_vector_index(
                [
                    {
                        "chunk_id": "attention",
                        "doc_id": "doc",
                        "source_path": "/docs/attention.md",
                        "file_type": "markdown",
                        "text": "注意力机制使用 Query、Key 和 Value。",
                    }
                ],
                provider=provider,
                path=Path(directory) / "vectors.jsonl",
            )
            chat = FakeToolChat()
            result = KnowledgeAgent(
                KnowledgeSearchTool(index, provider),
                chat,
                max_steps=3,
            ).run("注意力机制使用什么？")

            self.assertEqual(result.stopped_reason, "completed")
            self.assertEqual(chat.calls, 2)
            self.assertEqual(len(result.tool_calls), 1)
            self.assertEqual(result.tool_calls[0]["call_id"], "call-1")
            self.assertEqual(result.evidence[0]["chunk_id"], "attention")
            self.assertEqual(chat.tools[0][0]["name"], "search_knowledge_base")
            self.assertEqual(chat.messages[1][-1]["role"], "tool")
            self.assertEqual(result.history.size, 1)
            self.assertEqual(result.state.step, 2)
            self.assertEqual(result.state.stopped_reason, "completed")
            self.assertEqual(result.to_dict()["state"]["history"]["turns"][0]["user"], "注意力机制使用什么？")

    def test_agent_reuses_completed_history_without_persisting_tool_messages(self) -> None:
        class FinalOnlyChat:
            model = "fake-agent-model"

            def __init__(self) -> None:
                self.messages = []

            def complete_with_tools(self, messages, tools):
                self.messages.append(list(messages))
                return ToolChatTurn(
                    content="这是后续答案。",
                    tool_calls=(),
                    assistant_message={"role": "assistant", "content": "这是后续答案。"},
                )

        chat = FinalOnlyChat()
        agent = KnowledgeAgent(None, chat, max_steps=1, _system_prompt="系统规则")
        history = ConversationHistory().append("之前的问题", "之前的答案")
        result = agent.run("现在的问题", history=history)

        request = chat.messages[0]
        self.assertEqual(
            [(message["role"], message["content"]) for message in request],
            [
                ("system", "系统规则"),
                ("user", "之前的问题"),
                ("assistant", "之前的答案"),
                ("user", "现在的问题"),
            ],
        )
        self.assertEqual(result.history.size, 2)
        self.assertTrue(all(message["role"] != "tool" for message in result.history.to_messages()))
        self.assertEqual(result.state.to_dict()["step"], 1)

    def test_agent_stops_at_bound_and_returns_audit_result(self) -> None:
        class NeverFinishes(FakeToolChat):
            def complete_with_tools(self, messages, tools):
                self.calls += 1
                return ToolChatTurn(
                    content=None,
                    tool_calls=(
                        ToolCall("call", "search_knowledge_base", '{"query":"x"}'),
                    ),
                    assistant_message={
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [],
                    },
                )

        with tempfile.TemporaryDirectory() as directory:
            provider = HashEmbeddingProvider(dimension=64)
            index = build_vector_index(
                [
                    {
                        "chunk_id": "x",
                        "doc_id": "d",
                        "source_path": "/x.txt",
                        "file_type": "txt",
                        "text": "x",
                    }
                ],
                provider=provider,
                path=Path(directory) / "vectors.jsonl",
            )
            chat = NeverFinishes()
            result = KnowledgeAgent(
                KnowledgeSearchTool(index, provider), chat, max_steps=2
            ).run("x")
            self.assertEqual(result.stopped_reason, "max_steps")
            self.assertEqual(chat.calls, 2)
            self.assertIn("最大工具调用轮数", result.answer)
            self.assertEqual(result.history.size, 0)
            self.assertEqual(result.state.history.size, 0)


if __name__ == "__main__":
    unittest.main()
