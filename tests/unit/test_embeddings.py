from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rag_agent.embeddings import (
    ChineseNgramEmbeddingProvider,
    HashEmbeddingProvider,
    create_embedding_provider,
)
from rag_agent.embeddings.base import EmbeddingError
from rag_agent.retrieval import LocalVectorIndex, build_vector_index


class ChineseEmbeddingTests(unittest.TestCase):
    def test_chinese_provider_is_deterministic_and_normalizes_variants(self) -> None:
        provider = ChineseNgramEmbeddingProvider(dimension=128)
        first = provider.embed(["三次握手？"])[0]
        second = provider.embed(["三次握手?"])[0]
        self.assertEqual(first, provider.embed(["三次握手？"])[0])
        self.assertGreater(sum(left * right for left, right in zip(first, second)), 0.8)
        self.assertEqual(provider.name, "chinese")
        self.assertIn("chinese-ngram-v1", provider.fingerprint)

    def test_chinese_provider_retrieves_chinese_phrase(self) -> None:
        provider = ChineseNgramEmbeddingProvider(dimension=256)
        chunks = [
            {
                "chunk_id": "tcp",
                "doc_id": "tcp-doc",
                "source_path": "/docs/tcp.txt",
                "file_type": "txt",
                "text": "TCP 通过三次握手建立连接。",
            },
            {
                "chunk_id": "http",
                "doc_id": "http-doc",
                "source_path": "/docs/http.txt",
                "file_type": "txt",
                "text": "HTTP 使用请求和响应传输网页。",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            index = build_vector_index(
                chunks,
                provider=provider,
                path=Path(directory) / "vectors.jsonl",
            )
            results = index.search(
                "三次 握手如何建立连接？",
                provider=provider,
                top_k=1,
                min_score=-1,
            )
            self.assertEqual(results[0].chunk_id, "tcp")

    def test_factory_exposes_chinese_without_changing_hash_default(self) -> None:
        self.assertIsInstance(create_embedding_provider(), HashEmbeddingProvider)
        provider = create_embedding_provider("chinese", dimension=128)
        self.assertIsInstance(provider, ChineseNgramEmbeddingProvider)
        self.assertIsInstance(create_embedding_provider("zh"), ChineseNgramEmbeddingProvider)
        with self.assertRaises(EmbeddingError):
            # The provider name must be explicit in an index; using a hash
            # provider against a Chinese index is rejected by the existing guard.
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "vectors.jsonl"
                build_vector_index(
                    [
                        {
                            "chunk_id": "one",
                            "doc_id": "doc",
                            "source_path": "/docs/one.txt",
                            "file_type": "txt",
                            "text": "三次握手",
                        }
                    ],
                    provider=ChineseNgramEmbeddingProvider(dimension=128),
                    path=path,
                )
                LocalVectorIndex.load(path).search(
                    "三次握手",
                    provider=HashEmbeddingProvider(dimension=128),
                    top_k=1,
                )

    def test_invalid_dimension_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ChineseNgramEmbeddingProvider(dimension=32)
        with self.assertRaises(ValueError):
            ChineseNgramEmbeddingProvider(dimension=True)


if __name__ == "__main__":
    unittest.main()
