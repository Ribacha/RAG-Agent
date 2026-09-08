"""裁判模块的离线测试：解析契约、校准判读、交叉归因与 runner 集成。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rag_agent.harness import HarnessOptions, HarnessTask, run_harness
from rag_agent.harness.judge import (
    CALIBRATION_SAMPLES,
    JudgeError,
    calibrate_judge,
    cross_attribution,
    judge_task,
    parse_verdict,
)
from rag_agent.embeddings import HashEmbeddingProvider
from rag_agent.retrieval import build_vector_index


GOOD_VERDICT = (
    '{"faithfulness": 5, "coverage": 4, "overall": 4,'
    ' "failure_type": "正常", "reason": "证据支撑充分"}'
)


class FakeJudgeChat:
    """按 query 前缀返回预设判决的假裁判。"""

    model = "fake-judge"

    def __init__(self, default: str = GOOD_VERDICT) -> None:
        self.default = default
        self.prompts: list[str] = []

    def complete(self, messages):
        self.prompts.append(messages[-1]["content"])
        return self.default


TASK = HarnessTask(
    task_id="q1",
    query="三次握手？",
    task_type="answerable",
    relevant_chunk_ids=("c1",),
    expected_points=("SYN",),
)
PAYLOAD = {
    "answer": "三次握手建立连接 [1]。",
    "evidence": [{"source_path": "/docs/a.md", "text": "SYN SYN-ACK ACK", "source_type": "local"}],
    "tool_calls": [],
}


class ParseTests(unittest.TestCase):
    def test_parse_plain_and_fenced_json(self) -> None:
        verdict = parse_verdict(GOOD_VERDICT)
        self.assertEqual((verdict.faithfulness, verdict.coverage, verdict.overall), (5, 4, 4))
        fenced = f"裁判结果如下：\n```json\n{GOOD_VERDICT}\n```\n完毕"
        self.assertEqual(parse_verdict(fenced).failure_type, "正常")

    def test_contract_violations_rejected(self) -> None:
        for bad in (
            "不是 JSON",
            '{"faithfulness": 6, "coverage": 4, "overall": 4, "failure_type": "正常"}',
            '{"faithfulness": 5, "coverage": 4, "overall": 4, "failure_type": "别的"}',
            '{"faithfulness": true, "coverage": 4, "overall": 4, "failure_type": "正常"}',
        ):
            with self.assertRaises(JudgeError):
                parse_verdict(bad)


class JudgeTaskTests(unittest.TestCase):
    def test_judge_task_builds_expected_prompt_and_parses(self) -> None:
        chat = FakeJudgeChat()
        verdict = judge_task(TASK, PAYLOAD, chat)
        self.assertEqual(verdict.overall, 4)
        prompt = chat.prompts[0]
        self.assertIn("三次握手", prompt)
        self.assertIn("SYN", prompt)
        self.assertIn("（本地）/docs/a.md", prompt)
        self.assertNotIn("陷阱题", prompt)

    def test_refusal_task_gets_trap_instruction(self) -> None:
        refusal_task = HarnessTask(task_id="r1", query="火星改造？", task_type="refusal")
        chat = FakeJudgeChat()
        judge_task(refusal_task, PAYLOAD, chat)
        self.assertIn("陷阱题", chat.prompts[0])


class CalibrationTests(unittest.TestCase):
    def test_perfect_judge_no_warning(self) -> None:
        # 假裁判对每个校准样本返回与人工完全一致的判决
        by_query = {s["query"]: json.dumps(s["human"], ensure_ascii=False) for s in CALIBRATION_SAMPLES}

        class PerfectJudge:
            model = "perfect"

            def complete(self, messages):
                for query, verdict in by_query.items():
                    if query in messages[-1]["content"]:
                        return verdict
                raise AssertionError("未知校准样本")

        result = calibrate_judge(PerfectJudge())
        self.assertFalse(result["warning"])
        self.assertEqual(result["mean_overall_delta"], 0.0)
        self.assertEqual(result["type_mismatch_count"], 0)

    def test_biased_judge_triggers_warning(self) -> None:
        biased = json.dumps(
            {"faithfulness": 5, "coverage": 5, "overall": 5, "failure_type": "正常"},
            ensure_ascii=False,
        )
        result = calibrate_judge(FakeJudgeChat(default=biased))
        self.assertTrue(result["warning"])
        self.assertGreater(abs(result["mean_overall_delta"]), 1.0)
        self.assertGreater(result["type_mismatch_count"], 2)


class CrossAttributionTests(unittest.TestCase):
    def test_buckets_and_counts(self) -> None:
        rows = [
            {"task": {"id": "a"}, "metrics": {"retrieval_hit": True},
             "judged": {"overall": 4}},
            {"task": {"id": "b"}, "metrics": {"retrieval_hit": True},
             "judged": {"overall": 2}},   # 命中但低分 -> 生成端
            {"task": {"id": "c"}, "metrics": {"retrieval_hit": False},
             "judged": {"overall": 2}},   # 未命中且低分 -> 检索端
            {"task": {"id": "d"}, "metrics": {"retrieval_hit": False},
             "judged": {"overall": 4}},   # 未命中但高分 -> 自行补全
            {"task": {"id": "e"}, "metrics": {"retrieval_hit": None},
             "judged": {"overall": 5}},   # 无标注（refusal）
            {"task": {"id": "f"}, "metrics": {"retrieval_hit": True},
             "judged": {"error": "裁判失败"}},
        ]
        result = cross_attribution(rows)
        self.assertEqual(result["generation_side_failures"], 1)
        self.assertEqual(result["retrieval_side_failures"], 1)
        self.assertEqual(result["suspicious_self_answers"], 1)
        self.assertEqual(result["buckets"]["hit_and_high"], ["a"])
        self.assertEqual(result["buckets"]["unlabeled"], ["e"])
        self.assertIn("生成端", result["reading"])


class RunnerJudgeIntegrationTests(unittest.TestCase):
    def test_fake_run_with_fake_judge_produces_full_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "vectors.jsonl"
            build_vector_index(
                [{
                    "chunk_id": "tcp", "doc_id": "d",
                    "source_path": "/docs/net.md", "file_type": "markdown",
                    "text": "TCP 三次握手 SYN。",
                }],
                provider=HashEmbeddingProvider(dimension=64),
                path=index_path,
            )
            tasks_path = root / "tasks.jsonl"
            tasks_path.write_text(
                json.dumps({
                    "id": "q1", "query": "TCP 三次握手 SYN",
                    "relevant_chunk_ids": ["tcp"], "expected_points": ["SYN"],
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            report = run_harness(
                HarnessOptions(
                    tasks_path=tasks_path,
                    index_path=index_path,
                    mode="fake",
                    report_dir=root / "reports",
                ),
                judge_factory=FakeJudgeChat,
                calibrate=True,
            )
        self.assertEqual(report["judge_summary"]["judged_count"], 1)
        self.assertEqual(report["judge_summary"]["overall_mean"], 4.0)
        self.assertEqual(report["tasks"][0]["judged"]["failure_type"], "正常")
        self.assertEqual(report["cross_attribution"]["buckets"]["hit_and_high"], ["q1"])
        self.assertIsNotNone(report["judge_calibration"])
        self.assertEqual(report["judge_summary"]["judge_error_count"], 0)


if __name__ == "__main__":
    unittest.main()
