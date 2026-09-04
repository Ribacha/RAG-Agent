from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rag_agent.agent import KnowledgeSearchTool, KnowledgeToolError, SEARCH_KNOWLEDGE_TOOL
from rag_agent.embeddings import HashEmbeddingProvider
from rag_agent.retrieval import build_vector_index


class AgentToolTests(unittest.TestCase):
    def _tool(self) -> KnowledgeSearchTool:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        provider = HashEmbeddingProvider(dimension=64)
        index = build_vector_index(
            [
                {
                    "chunk_id": "one",
                    "doc_id": "doc",
                    "source_path": "/docs/a.md",
                    "file_type": "markdown",
                    "text": "Query Key Value 属于注意力机制。",
                }
            ],
            provider=provider,
            path=Path(directory.name) / "vectors.jsonl",
        )
        return KnowledgeSearchTool(index, provider)

    def test_schema_is_strict_and_does_not_expose_paths(self) -> None:
        self.assertEqual(SEARCH_KNOWLEDGE_TOOL["name"], "search_knowledge_base")
        parameters = SEARCH_KNOWLEDGE_TOOL["parameters"]
        self.assertTrue(parameters["additionalProperties"] is False)
        self.assertNotIn("path", parameters["properties"])

    def test_invoke_returns_source_aware_json(self) -> None:
        result = self._tool().invoke("Query Key Value", top_k=1, min_score=-1)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["source_path"], "/docs/a.md")
        self.assertEqual(result["results"][0]["chunk_id"], "one")

    def test_invoke_json_rejects_unknown_or_invalid_arguments(self) -> None:
        tool = self._tool()
        with self.assertRaises(KnowledgeToolError):
            tool.invoke_json('{"query":"x","path":"/etc/passwd"}')
        with self.assertRaises(KnowledgeToolError):
            tool.invoke(" ")
        with self.assertRaises(KnowledgeToolError):
            tool.invoke("x", top_k=21)
        with self.assertRaises(KnowledgeToolError):
            tool.invoke_json("[]")

    def test_invoke_json_round_trip(self) -> None:
        result = self._tool().invoke_json('{"query":"注意力","top_k":1}')
        self.assertIn('"query": "注意力"', result)
        self.assertIn('"count": 1', result)


if __name__ == "__main__":
    unittest.main()
