from __future__ import annotations

from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

from rag_agent.agent import (
    ConversationHistory,
    GraphUnavailableError,
    KnowledgeAgent,
    agent_model_node,
    agent_result_from_graph_state,
    agent_tools_node,
    build_knowledge_graph,
    finalize_graph_node,
    initial_graph_state,
    langgraph_available,
    route_after_agent,
    route_after_tools,
)
from rag_agent.answering.chat import ToolCall, ToolChatTurn
from rag_agent.embeddings import HashEmbeddingProvider
from rag_agent.retrieval import build_vector_index


class GraphChat:
    model = "fake-graph-model"

    def __init__(self) -> None:
        self.calls = 0
        self.messages = []

    def complete_with_tools(self, messages, tools):
        self.calls += 1
        self.messages.append(list(messages))
        if self.calls == 1:
            arguments = '{"query":"注意力","top_k":1,"min_score":-1}'
            return ToolChatTurn(
                content=None,
                tool_calls=(ToolCall("graph-call", "search_knowledge_base", arguments),),
                assistant_message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "graph-call",
                            "type": "function",
                            "function": {
                                "name": "search_knowledge_base",
                                "arguments": arguments,
                            },
                        }
                    ],
                },
            )
        return ToolChatTurn(
            content="图状态回答。[1]",
            tool_calls=(),
            assistant_message={"role": "assistant", "content": "图状态回答。[1]"},
        )


class GraphTests(unittest.TestCase):
    def _agent(self, directory: str) -> KnowledgeAgent:
        provider = HashEmbeddingProvider(dimension=64)
        index = build_vector_index(
            [
                {
                    "chunk_id": "attention",
                    "doc_id": "doc",
                    "source_path": "/docs/attention.txt",
                    "file_type": "txt",
                    "text": "注意力机制使用 Query、Key 和 Value。",
                }
            ],
            provider=provider,
            path=Path(directory) / "vectors.jsonl",
        )
        from rag_agent.agent import KnowledgeSearchTool

        return KnowledgeAgent(KnowledgeSearchTool(index, provider), GraphChat(), max_steps=3)

    def test_nodes_follow_agent_tools_finalize_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(directory)
            initial = initial_graph_state(
                agent,
                "注意力是什么？",
                history=ConversationHistory().append("旧问题", "旧答案"),
            )
            first = agent_model_node(agent, initial)
            self.assertEqual(first["step"], 1)
            self.assertEqual(route_after_agent(first), "tools")
            after_tools = agent_tools_node(agent, {**initial, **first})
            self.assertEqual(after_tools["pending_tool_calls"], [])
            self.assertEqual(after_tools["evidence"][0]["chunk_id"], "attention")
            second = agent_model_node(agent, {**initial, **first, **after_tools})
            self.assertEqual(route_after_agent(second), "finalize")
            final = finalize_graph_node({**initial, **first, **after_tools, **second})
            state = {**initial, **first, **after_tools, **second, **final}
            result = agent_result_from_graph_state(agent, state)

            self.assertEqual(result.answer, "图状态回答。[1]")
            self.assertEqual(result.stopped_reason, "completed")
            self.assertEqual(result.history.size, 2)
            self.assertEqual(result.history.turns[-1].user, "注意力是什么？")
            self.assertEqual(result.state.step, 2)
            self.assertEqual(result.state.messages[-1]["role"], "assistant")

    def test_max_step_tool_turn_routes_to_finalize_without_history_pollution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(directory)
            agent.max_steps = 1
            initial = initial_graph_state(agent, "注意力是什么？")
            first = agent_model_node(agent, initial)
            self.assertEqual(route_after_agent(first), "tools")
            after_tools = agent_tools_node(agent, {**initial, **first})
            self.assertEqual(route_after_tools(agent, {**initial, **first, **after_tools}), "finalize")
            final = finalize_graph_node({**initial, **first, **after_tools})
            result = agent_result_from_graph_state(
                agent, {**initial, **first, **after_tools, **final}
            )
            self.assertEqual(result.stopped_reason, "max_steps")
            self.assertEqual(result.history.size, 0)

    def test_graph_dependency_is_lazy_and_reports_install_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(directory)
            if langgraph_available():
                graph = build_knowledge_graph(agent)
                self.assertTrue(hasattr(graph, "invoke"))
            else:
                with self.assertRaisesRegex(GraphUnavailableError, "LangGraph 未安装"):
                    build_knowledge_graph(agent)

    def test_builder_registers_observable_nodes_and_routes(self) -> None:
        class FakeCompiledGraph:
            def invoke(self, state, config=None):
                return state

        class FakeStateGraph:
            last = None

            def __init__(self, schema):
                self.schema = schema
                self.nodes = {}
                self.routes = []
                self.entry = None
                self.finish = None
                FakeStateGraph.last = self

            def add_node(self, name, node):
                self.nodes[name] = node

            def add_conditional_edges(self, source, router, mapping):
                self.routes.append((source, router, mapping))

            def set_entry_point(self, name):
                self.entry = name

            def set_finish_point(self, name):
                self.finish = name

            def compile(self):
                return FakeCompiledGraph()

        fake_package = types.ModuleType("langgraph")
        fake_graph = types.ModuleType("langgraph.graph")
        fake_graph.StateGraph = FakeStateGraph
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(directory)
            with patch.dict(
                sys.modules,
                {"langgraph": fake_package, "langgraph.graph": fake_graph},
            ):
                compiled = build_knowledge_graph(agent)
            self.assertTrue(hasattr(compiled, "invoke"))
            self.assertEqual(set(FakeStateGraph.last.nodes), {"agent", "tools", "finalize"})
            self.assertEqual(FakeStateGraph.last.entry, "agent")
            self.assertEqual(FakeStateGraph.last.finish, "finalize")
            self.assertEqual(
                [route[0] for route in FakeStateGraph.last.routes],
                ["agent", "tools"],
            )


if __name__ == "__main__":
    unittest.main()
