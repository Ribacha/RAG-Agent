from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rag_agent.answering import RagAnswerer, build_evidence_context
from rag_agent.embeddings import HashEmbeddingProvider
from rag_agent.retrieval import build_vector_index


class FakeChat:
    model = "fake-chat"

    def __init__(self, answer: str = "根据资料可知。[2]") -> None:
        self.answer = answer
        self.messages: list[dict[str, str]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.messages = messages
        return self.answer


class AnsweringTests(unittest.TestCase):
    def _make_index(self, directory: str):
        chunks = [
            {
                "chunk_id": "first",
                "doc_id": "doc-1",
                "source_path": "/docs/one.md",
                "file_type": "markdown",
                "text": "Transformer 使用注意力机制。",
                "heading_path": ["Transformer"],
            },
            {
                "chunk_id": "second",
                "doc_id": "doc-2",
                "source_path": "/docs/two.pdf",
                "file_type": "pdf",
                "text": "注意力计算会使用 Query、Key 和 Value。",
                "page_start": 7,
                "page_end": 7,
                "heading_path": [],
            },
        ]
        provider = HashEmbeddingProvider(dimension=96)
        index = build_vector_index(
            chunks,
            provider=provider,
            path=Path(directory) / "vectors.jsonl",
        )
        return index, provider

    def test_answer_passes_bounded_evidence_and_tracks_citation_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index, provider = self._make_index(directory)
            chat = FakeChat("结论见第二条资料。[2]")
            result = RagAnswerer(
                index,
                embedding_provider=provider,
                chat_provider=chat,
                min_score=-1,
                max_context_chars=500,
                top_k=2,
            ).answer("Query Key Value 如何参与注意力计算？")

            self.assertEqual(result.used_model, "fake-chat")
            self.assertEqual(
                [item.chunk_id for item in result.citations],
                [result.results[1].chunk_id],
            )
            self.assertIn("<evidence>", chat.messages[1]["content"])
            self.assertLessEqual(
                len(build_evidence_context(result.results, 500)),
                500,
            )

    def test_no_evidence_does_not_call_chat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index, provider = self._make_index(directory)
            chat = FakeChat()
            result = RagAnswerer(
                index,
                embedding_provider=provider,
                chat_provider=chat,
                min_score=1.1,
            ).answer("一个完全不相关的问题")

            self.assertIn("没有找到足够", result.answer)
            self.assertEqual(chat.messages, [])

    def test_without_chat_provider_is_an_explicit_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index, provider = self._make_index(directory)
            result = RagAnswerer(
                index,
                embedding_provider=provider,
                chat_provider=None,
                min_score=-1,
                top_k=1,
            ).answer("注意力")

            self.assertIsNone(result.used_model)
            self.assertIn("来源：", result.answer)
            self.assertEqual(len(result.citations), 1)


if __name__ == "__main__":
    unittest.main()
