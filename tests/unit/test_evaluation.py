from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from rag_agent.cli import main
from rag_agent.embeddings import HashEmbeddingProvider
from rag_agent.evaluation import (
    EvaluationSample,
    evaluate,
    load_evaluation_samples,
)
from rag_agent.retrieval import build_vector_index


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            {
                "chunk_id": "chunk-attention",
                "doc_id": "doc-a",
                "source_path": "/docs/transformer.md",
                "file_type": "markdown",
                "text": "Transformer 的 self attention 使用 Query、Key 和 Value。",
            },
            {
                "chunk_id": "chunk-network",
                "doc_id": "doc-b",
                "source_path": "/docs/network.txt",
                "file_type": "txt",
                "text": "TCP 通过三次握手建立连接。",
            },
        ]

    def test_evaluate_reports_chunk_hit_and_source_hit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = build_vector_index(
                self.chunks,
                provider=HashEmbeddingProvider(dimension=96),
                path=Path(directory) / "vectors.jsonl",
            )
            report = evaluate(
                index,
                [
                    EvaluationSample(
                        query="Query Key Value 的关系",
                        relevant_chunk_ids=("chunk-attention",),
                        name="attention",
                    ),
                    EvaluationSample(
                        query="TCP 握手",
                        relevant_source_paths=("/docs/network.txt",),
                        name="network",
                    ),
                ],
                provider=HashEmbeddingProvider(dimension=96),
                top_k=1,
                min_score=-1,
            )

            self.assertEqual(report.sample_count, 2)
            self.assertEqual(report.recall_at_k, 1.0)
            self.assertEqual(report.citation_accuracy, 1.0)
            self.assertEqual(report.samples[0].matched_chunk_ids, ("chunk-attention",))
            self.assertEqual(report.samples[1].matched_source_paths, ("/docs/network.txt",))
            self.assertEqual(report.to_dict()["samples"][0]["retrieved_chunk_ids"], ["chunk-attention"])

    def test_citation_proxy_counts_relevant_results_and_sample_threshold_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = build_vector_index(
                self.chunks,
                provider=HashEmbeddingProvider(dimension=96),
                path=Path(directory) / "vectors.jsonl",
            )
            report = evaluate(
                index,
                [
                    EvaluationSample(
                        query="Query Key Value",
                        relevant_chunk_ids=("chunk-attention",),
                        min_score=-1,
                    )
                ],
                provider=HashEmbeddingProvider(dimension=96),
                top_k=2,
                min_score=2,
            )
            self.assertEqual(report.samples[0].min_score, -1.0)
            self.assertEqual(report.samples[0].recall_at_k, 1.0)
            self.assertGreaterEqual(report.samples[0].citation_accuracy, 0.0)
            self.assertLessEqual(report.samples[0].citation_accuracy, 1.0)

    def test_empty_labels_and_invalid_jsonl_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "至少需要"):
            EvaluationSample(query="test")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.jsonl"
            path.write_text(
                json.dumps({"query": "test", "relevant_chunk_ids": []}, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "第 1 行"):
                load_evaluation_samples(path)

            path.write_text(
                json.dumps({"query": "test", "relevant_chunk_ids": "chunk"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "字符串数组"):
                load_evaluation_samples(path)

    def test_cli_evaluate_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "vectors.jsonl"
            eval_path = root / "eval.jsonl"
            build_vector_index(
                self.chunks,
                provider=HashEmbeddingProvider(dimension=96),
                path=index_path,
            )
            eval_path.write_text(
                json.dumps(
                    {
                        "id": "attention",
                        "query": "Query Key Value",
                        "relevant_chunk_ids": ["chunk-attention"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(
                    [
                        "evaluate",
                        str(eval_path),
                        "--index",
                        str(index_path),
                        "--embedding-dimension",
                        "96",
                        "--top-k",
                        "1",
                        "--min-score",
                        "-1",
                        "--json",
                    ]
                )
            self.assertEqual(exit_code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["sample_count"], 1)
            self.assertEqual(payload["recall_at_k"], 1.0)
            self.assertEqual(payload["samples"][0]["name"], "attention")


if __name__ == "__main__":
    unittest.main()
