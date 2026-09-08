"""Harness 跑批器的离线端到端测试（fake 模式：理想剧本应得满分）。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rag_agent.harness import HarnessOptions, compare_reports, run_harness
from rag_agent.embeddings import HashEmbeddingProvider
from rag_agent.retrieval import build_vector_index


def _workspace(testcase: unittest.TestCase) -> tuple[Path, Path, Path]:
    directory = tempfile.TemporaryDirectory()
    testcase.addCleanup(directory.cleanup)
    root = Path(directory.name)
    index_path = root / "vectors.jsonl"
    build_vector_index(
        [
            {
                "chunk_id": "tcp",
                "doc_id": "d1",
                "source_path": "/docs/net.md",
                "file_type": "markdown",
                "text": "TCP 通过三次握手建立连接，需要 SYN SYN-ACK ACK。",
            }
        ],
        provider=HashEmbeddingProvider(dimension=64),
        path=index_path,
    )
    tasks_path = root / "tasks.jsonl"
    tasks_path.write_text(
        json.dumps(
            {
                "id": "q1",
                "query": "TCP 三次握手建立连接 SYN",
                "task_type": "answerable",
                "relevant_chunk_ids": ["tcp"],
                "expected_points": ["SYN"],
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps(
            {
                "id": "r1",
                "query": "量子纠缠通信在哪一层",
                "task_type": "refusal",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    reports = root / "reports"
    return tasks_path, index_path, reports


class FakeModeEndToEndTests(unittest.TestCase):
    def test_ideal_script_scores_full_marks(self) -> None:
        tasks_path, index_path, reports = _workspace(self)
        report = run_harness(
            HarnessOptions(
                tasks_path=tasks_path,
                index_path=index_path,
                mode="fake",
                report_dir=reports,
            )
        )
        self.assertEqual(report["mode"], "fake")
        self.assertEqual(report["model"], "ideal-script")
        answerable = report["aggregate"]["by_type"]["answerable"]
        refusal = report["aggregate"]["by_type"]["refusal"]
        # 尺子自证：理想剧本在健康索引上必须满分
        self.assertEqual(answerable["retrieval_hit_rate"], 1.0)
        self.assertEqual(answerable["refusal_correct_rate"], 1.0)
        self.assertEqual(answerable["citation_ok_rate"], 1.0)
        self.assertEqual(refusal["refusal_correct_rate"], 1.0)
        self.assertFalse(report["tasks"][0]["result"]["web_tool_enabled"])
        # 报告已落盘且可解析
        report_path = Path(report["report_path"])
        self.assertTrue(report_path.exists())
        self.assertEqual(
            json.loads(report_path.read_text(encoding="utf-8"))["schema_version"], 1
        )

    def test_repeat_produces_jitter_section(self) -> None:
        tasks_path, index_path, reports = _workspace(self)
        report = run_harness(
            HarnessOptions(
                tasks_path=tasks_path,
                index_path=index_path,
                mode="fake",
                repeat=3,
                report_dir=reports,
            )
        )
        jitter = report["jitter"]
        self.assertIsNotNone(jitter)
        self.assertEqual(jitter["run_count"], 3)
        self.assertEqual(jitter["refusal_flip_count"], 0)  # 确定性模型零翻转
        self.assertEqual(jitter["refusal_correct_rate"]["std"], 0.0)

    def test_compare_reports_detects_flips_and_deltas(self) -> None:
        tasks_path, index_path, reports = _workspace(self)
        first = run_harness(
            HarnessOptions(tasks_path=tasks_path, index_path=index_path, report_dir=reports)
        )
        second = run_harness(
            HarnessOptions(
                tasks_path=tasks_path,
                index_path=index_path,
                report_dir=reports,
                compare_path=Path(first["report_path"]),
            )
        )
        compare = second["compare"]
        self.assertIn("refusal_correct_rate", compare["metric_deltas"])
        self.assertEqual(compare["metric_deltas"]["refusal_correct_rate"]["new"], 1.0)
        self.assertEqual(compare["refusal_flips_between_runs"], [])

        # 人为构造一份旧报告制造翻转，验证对比逻辑
        tampered = json.loads(Path(first["report_path"]).read_text(encoding="utf-8"))
        tampered["tasks"][0]["metrics"]["refused"] = True
        flip_compare = compare_reports(tampered, second)
        self.assertEqual(flip_compare["refusal_flips_between_runs"], ["q1"])

    def test_limit_and_real_mode_validation(self) -> None:
        tasks_path, index_path, reports = _workspace(self)
        report = run_harness(
            HarnessOptions(
                tasks_path=tasks_path,
                index_path=index_path,
                limit=1,
                report_dir=reports,
            )
        )
        self.assertEqual(report["task_count"], 1)
        with self.assertRaises(ValueError):
            run_harness(
                HarnessOptions(
                    tasks_path=tasks_path,
                    index_path=index_path,
                    mode="real",  # real 无 chat_factory 必须显式失败
                    report_dir=reports,
                )
            )


if __name__ == "__main__":
    unittest.main()
