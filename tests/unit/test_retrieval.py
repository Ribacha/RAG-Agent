from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rag_agent.embeddings import HashEmbeddingProvider
from rag_agent.embeddings.base import EmbeddingError
from rag_agent.retrieval import LocalVectorIndex, build_vector_index, update_vector_index


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            {
                "chunk_id": "chunk-attention",
                "doc_id": "doc-a",
                "source_path": "/docs/transformer.md",
                "file_type": "markdown",
                "text": "Transformer 的 self attention 使用 Query、Key 和 Value。",
                "heading_path": ["Transformer", "Attention"],
                "page_start": None,
                "page_end": None,
            },
            {
                "chunk_id": "chunk-network",
                "doc_id": "doc-b",
                "source_path": "/docs/network.txt",
                "file_type": "txt",
                "text": "TCP 通过三次握手建立连接。",
                "heading_path": [],
                "page_start": None,
                "page_end": None,
            },
        ]

    def test_build_and_search_preserves_citation_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.jsonl"
            provider = HashEmbeddingProvider(dimension=96)
            index = build_vector_index(self.chunks, provider=provider, path=path)

            results = index.search(
                "Query Key Value 的关系",
                provider=provider,
                top_k=1,
                min_score=-1,
            )

            self.assertEqual(index.size, 2)
            self.assertEqual(results[0].chunk_id, "chunk-attention")
            self.assertEqual(results[0].heading_path, ("Transformer", "Attention"))
            self.assertGreater(results[0].score, -1)

            loaded = LocalVectorIndex.load(path)
            self.assertEqual(loaded.size, 2)
            self.assertEqual(
                loaded.search("TCP 握手", provider=provider, top_k=1, min_score=-1)[0].chunk_id,
                "chunk-network",
            )

    def test_provider_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.jsonl"
            build_vector_index(
                self.chunks,
                provider=HashEmbeddingProvider(dimension=64),
                path=path,
            )
            index = LocalVectorIndex.load(path)
            with self.assertRaises(EmbeddingError):
                index.search(
                    "attention",
                    provider=HashEmbeddingProvider(dimension=128),
                )

    def test_empty_query_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.jsonl"
            provider = HashEmbeddingProvider(dimension=64)
            index = build_vector_index(self.chunks, provider=provider, path=path)

            self.assertEqual(index.search("  ", provider=provider), [])
            results = index.search(
                "TCP",
                provider=provider,
                top_k=5,
                min_score=0.2,
                file_type="markdown",
            )
            self.assertEqual(results, [])

    def test_index_file_has_inspectable_meta_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.jsonl"
            build_vector_index(
                self.chunks,
                provider=HashEmbeddingProvider(dimension=64),
                path=path,
            )
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["_type"], "meta")
            self.assertEqual(rows[0]["chunk_count"], 2)
            self.assertEqual(sum(row["_type"] == "chunk" for row in rows), 2)

    def test_incremental_update_reuses_unchanged_vectors(self) -> None:
        class CountingProvider:
            name = "hash"
            model = "hash-v1"
            dimension = 64

            def __init__(self):
                self.batches = []
                self._delegate = HashEmbeddingProvider(dimension=64)

            @property
            def fingerprint(self):
                return self._delegate.fingerprint

            def embed(self, texts):
                self.batches.append(list(texts))
                return self._delegate.embed(texts)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.jsonl"
            provider = CountingProvider()
            first = build_vector_index(self.chunks, provider=provider, path=path)
            self.assertEqual(len(provider.batches[-1]), 2)
            changed = [dict(row) for row in self.chunks]
            changed[1]["text"] = "TCP 通过四次握手建立连接。"
            changed[1]["content_hash"] = "changed"
            updated, stats = update_vector_index(changed, provider=provider, path=path)

            self.assertEqual(stats.reused_vectors, 1)
            self.assertEqual(stats.embedded_vectors, 1)
            self.assertEqual(len(provider.batches[-1]), 1)
            self.assertEqual(updated.size, 2)
            self.assertEqual(first.search("attention", provider=provider, min_score=-1)[0].chunk_id, "chunk-attention")

    def test_incremental_update_rebuilds_when_provider_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.jsonl"
            build_vector_index(self.chunks, provider=HashEmbeddingProvider(dimension=64), path=path)
            provider = HashEmbeddingProvider(dimension=96)
            _, stats = update_vector_index(self.chunks, provider=provider, path=path)
            self.assertEqual(stats.reused_vectors, 0)
            self.assertEqual(stats.embedded_vectors, 2)


if __name__ == "__main__":
    unittest.main()
