from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from rag_agent.cli import main


class CliRagIntegrationTests(unittest.TestCase):
    """Exercise the complete offline CLI path without a chat API key."""

    def test_ingest_search_evaluate_and_dry_run_form_a_grounded_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inbox = root / "inbox"
            output = root / "out"
            inbox.mkdir()
            (inbox / "guide.md").write_text(
                "# Guide\n\n本项目使用本地 embedding 建立可检索的 JSONL 索引。\n",
                encoding="utf-8",
            )
            plan = inbox / "plan.md"
            plan.write_text(
                "# Evaluation\n\n"
                "Recall@K 表示 Top-K 中命中相关证据的比例；"
                "citation_accuracy 是检索层引用正确率代理。\n",
                encoding="utf-8",
            )

            ingest_output = io.StringIO()
            with contextlib.redirect_stdout(ingest_output):
                self.assertEqual(
                    main(
                        [
                            "ingest",
                            str(inbox),
                            "--output",
                            str(output / "chunks.jsonl"),
                            "--manifest",
                            str(output / "documents.jsonl"),
                            "--failures",
                            str(output / "failures.jsonl"),
                            "--index",
                            str(output / "vectors.jsonl"),
                            "--embedding-provider",
                            "chinese",
                            "--embedding-dimension",
                            "128",
                        ]
                    ),
                    0,
                )
            ingest_summary = json.loads(ingest_output.getvalue())
            self.assertEqual(ingest_summary["documents_succeeded"], 2)
            self.assertEqual(ingest_summary["documents_failed"], 0)

            search_output = io.StringIO()
            with contextlib.redirect_stdout(search_output):
                self.assertEqual(
                    main(
                        [
                            "search",
                            "Recall@K 如何计算",
                            "--index",
                            str(output / "vectors.jsonl"),
                            "--embedding-provider",
                            "chinese",
                            "--embedding-dimension",
                            "128",
                            "--top-k",
                            "5",
                            "--min-score",
                            "0.08",
                            "--json",
                        ]
                    ),
                    0,
                )
            search_results = json.loads(search_output.getvalue())
            self.assertTrue(search_results)
            self.assertIn(
                str(plan.resolve()),
                {result["source_path"] for result in search_results},
            )

            evaluation = root / "evaluation.jsonl"
            evaluation.write_text(
                json.dumps(
                    {
                        "name": "recall",
                        "query": "Recall@K 如何计算",
                        "relevant_source_paths": [str(plan.resolve())],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            evaluate_output = io.StringIO()
            with contextlib.redirect_stdout(evaluate_output):
                self.assertEqual(
                    main(
                        [
                            "evaluate",
                            str(evaluation),
                            "--index",
                            str(output / "vectors.jsonl"),
                            "--embedding-provider",
                            "chinese",
                            "--embedding-dimension",
                            "128",
                            "--top-k",
                            "5",
                            "--min-score",
                            "0.08",
                            "--json",
                        ]
                    ),
                    0,
                )
            report = json.loads(evaluate_output.getvalue())
            self.assertEqual(report["sample_count"], 1)
            self.assertEqual(report["recall_at_k"], 1.0)

            dry_run_output = io.StringIO()
            with contextlib.redirect_stdout(dry_run_output):
                self.assertEqual(
                    main(
                        [
                            "ask",
                            "Recall@K 如何计算",
                            "--index",
                            str(output / "vectors.jsonl"),
                            "--embedding-provider",
                            "chinese",
                            "--embedding-dimension",
                            "128",
                            "--top-k",
                            "5",
                            "--min-score",
                            "0.08",
                            "--dry-run",
                            "--json",
                        ]
                    ),
                    0,
                )
            dry_run = json.loads(dry_run_output.getvalue())
            self.assertIsNone(dry_run["used_model"])
            self.assertIn(str(plan.resolve()), dry_run["answer"])
            self.assertIn("Recall@K", dry_run["answer"])

            no_evidence_output = io.StringIO()
            with contextlib.redirect_stdout(no_evidence_output):
                self.assertEqual(
                    main(
                        [
                            "ask",
                            "量子泡沫飞船的紫色燃料配方",
                            "--index",
                            str(output / "vectors.jsonl"),
                            "--embedding-provider",
                            "chinese",
                            "--embedding-dimension",
                            "128",
                            "--top-k",
                            "3",
                            "--min-score",
                            "0.5",
                            "--dry-run",
                            "--json",
                        ]
                    ),
                    0,
                )
            no_evidence = json.loads(no_evidence_output.getvalue())
            self.assertEqual(no_evidence["results"], [])
            self.assertIn("没有找到足够相关", no_evidence["answer"])


if __name__ == "__main__":
    unittest.main()
