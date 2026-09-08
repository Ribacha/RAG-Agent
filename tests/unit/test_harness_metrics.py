"""题库加载与确定性指标的离线单元测试。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rag_agent.answering.rag import DEFAULT_NO_EVIDENCE
from rag_agent.harness import (
    CitationStatus,
    HarnessTask,
    evaluate_task,
    load_tasks,
    refusal_detected,
)
from rag_agent.harness.metrics import aggregate_metrics


def write_tasks(testcase: unittest.TestCase, rows: list[dict]) -> Path:
    directory = tempfile.TemporaryDirectory()
    testcase.addCleanup(directory.cleanup)
    path = Path(directory.name) / "tasks.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def make_result(
    *,
    answer: str = "结论 [1]。",
    evidence: list[dict] | None = None,
    tool_calls: list[dict] | None = None,
) -> dict:
    if evidence is None:
        evidence = [{"chunk_id": "c1", "source_path": "/docs/a.md", "source_type": "local"}]
    if tool_calls is None:
        tool_calls = [
            {
                "name": "search_knowledge_base",
                "result": {"results": [{"chunk_id": "c1", "source_path": "/docs/a.md"}]},
            }
        ]
    return {"answer": answer, "evidence": evidence, "tool_calls": tool_calls}


ANSWERABLE = HarnessTask(
    task_id="q001",
    query="三次握手？",
    task_type="answerable",
    relevant_chunk_ids=("c1",),
    expected_points=("SYN",),
)
REFUSAL_TASK = HarnessTask(task_id="r001", query="火星改造？", task_type="refusal")


class TaskLoadingTests(unittest.TestCase):
    def test_valid_rows_parse_with_defaults(self) -> None:
        path = write_tasks(
            self,
            [
                {
                    "id": "q1",
                    "query": "问题",
                    "relevant_chunk_ids": ["c"],
                    "expected_points": ["要点"],
                }
            ]
        )
        tasks = load_tasks(path)
        self.assertEqual(tasks[0].task_type, "answerable")
        self.assertFalse(tasks[0].auto)
        self.assertEqual(tasks[0].expected_points, ("要点",))

    def test_validation_errors_carry_line_numbers(self) -> None:
        cases = [
            {"id": "", "query": "q", "expected_points": ["x"]},
            {"id": "q", "query": "  ", "expected_points": ["x"]},
            {"id": "q", "query": "q", "task_type": "open", "expected_points": ["x"]},
            {"id": "q", "query": "q", "expected_points": ["x"]},  # 缺标注
            {"id": "q", "query": "q", "relevant_chunk_ids": ["c"]},  # 缺要点
            {"id": "q", "query": "q", "relevant_chunk_ids": ["c"], "expected_points": ["x"] * 9},
        ]
        for row in cases:
            with self.assertRaises(ValueError) as raised:
                load_tasks(write_tasks(self, [row]))
            self.assertIn("第 1 行", str(raised.exception))

    def test_duplicate_ids_rejected(self) -> None:
        row = {"id": "q", "query": "q", "relevant_chunk_ids": ["c"], "expected_points": ["x"]}
        with self.assertRaises(ValueError) as raised:
            load_tasks(write_tasks(self, [row, dict(row)]))
        self.assertIn("id 重复", str(raised.exception))

    def test_refusal_task_needs_no_labels(self) -> None:
        tasks = load_tasks(
            write_tasks(
                self,
                [{"id": "r1", "query": "陷阱", "task_type": "refusal"}]
            )
        )
        self.assertEqual(tasks[0].task_type, "refusal")
        self.assertEqual(tasks[0].expected_points, ())


class MetricsTests(unittest.TestCase):
    def test_refusal_detection_variants(self) -> None:
        self.assertTrue(refusal_detected(DEFAULT_NO_EVIDENCE, 0))
        self.assertTrue(refusal_detected("知识库中没有找到足够相关的资料。", 0))
        self.assertFalse(refusal_detected("知识库中没有找到足够相关的资料。", 3))
        self.assertFalse(refusal_detected("答案是三次握手 [1]。", 1))

    def test_answerable_task_with_valid_citation(self) -> None:
        metric = evaluate_task(ANSWERABLE, make_result())
        self.assertTrue(metric.retrieval_hit)
        self.assertEqual(metric.citation_status, CitationStatus.OK)
        self.assertFalse(metric.refused)
        self.assertTrue(metric.refusal_correct)
        self.assertEqual(metric.tool_calls, 1)
        self.assertFalse(metric.web_used)

    def test_citation_status_branches(self) -> None:
        # 引用越界编号
        self.assertEqual(
            evaluate_task(ANSWERABLE, make_result(answer="结论 [7]。")).citation_status,
            CitationStatus.INVALID,
        )
        # 有结论但零引用
        self.assertEqual(
            evaluate_task(ANSWERABLE, make_result(answer="结论是没有引用的。")).citation_status,
            CitationStatus.UNCITED,
        )
        # 无证据（也就无从引用）
        self.assertEqual(
            evaluate_task(
                ANSWERABLE,
                make_result(answer="一些话", evidence=[], tool_calls=[]),
            ).citation_status,
            CitationStatus.NOT_APPLICABLE,
        )

    def test_retrieval_miss_when_labels_do_not_match(self) -> None:
        result = make_result(
            tool_calls=[
                {
                    "name": "search_knowledge_base",
                    "result": {"results": [{"chunk_id": "other", "source_path": "/x"}]},
                }
            ]
        )
        self.assertFalse(evaluate_task(ANSWERABLE, result).retrieval_hit)

    def test_refusal_task_correctness_both_directions(self) -> None:
        refused = evaluate_task(
            REFUSAL_TASK, make_result(answer=DEFAULT_NO_EVIDENCE, evidence=[], tool_calls=[])
        )
        self.assertTrue(refused.refused)
        self.assertTrue(refused.refusal_correct)
        wrong_answer = evaluate_task(
            REFUSAL_TASK, make_result(answer="火星改造的步骤如下 [1]。")
        )
        self.assertFalse(wrong_answer.refused)
        self.assertFalse(wrong_answer.refusal_correct)
        # answerable 题误拒同样算错
        wrongly_refused = evaluate_task(
            ANSWERABLE, make_result(answer=DEFAULT_NO_EVIDENCE, evidence=[], tool_calls=[])
        )
        self.assertFalse(wrongly_refused.refusal_correct)

    def test_web_used_counts_fetch_attempts(self) -> None:
        result = make_result(
            tool_calls=[
                {"name": "search_knowledge_base", "result": {"results": []}},
                {"name": "fetch_web_page", "result": {"error": "抓取上限"}},
            ]
        )
        metric = evaluate_task(ANSWERABLE, result)
        self.assertTrue(metric.web_used)
        self.assertEqual(metric.tool_calls, 2)

    def test_aggregate_splits_by_type_and_provenance(self) -> None:
        rows = [
            (ANSWERABLE, evaluate_task(ANSWERABLE, make_result())),
            (
                HarnessTask(
                    task_id="q2",
                    query="q",
                    task_type="answerable",
                    relevant_chunk_ids=("c1",),
                    expected_points=("x",),
                    category="传输层",
                    auto=True,
                ),
                evaluate_task(
                    ANSWERABLE,
                    make_result(
                        answer=DEFAULT_NO_EVIDENCE, evidence=[], tool_calls=[]
                    ),
                ),
            ),
            (REFUSAL_TASK, evaluate_task(REFUSAL_TASK, make_result(
                answer=DEFAULT_NO_EVIDENCE, evidence=[], tool_calls=[]
            ))),
        ]
        summary = aggregate_metrics(rows)
        self.assertEqual(summary["overall"]["count"], 3)
        self.assertEqual(summary["by_type"]["answerable"]["count"], 2)
        self.assertEqual(summary["by_type"]["refusal"]["count"], 1)
        self.assertEqual(summary["by_provenance"]["human"]["count"], 2)
        self.assertEqual(summary["by_provenance"]["auto"]["count"], 1)
        self.assertEqual(summary["by_category"]["传输层"]["count"], 1)
        # answerable 两题：一题命中一题误拒
        self.assertEqual(summary["by_type"]["answerable"]["retrieval_hit_rate"], 0.5)
        self.assertEqual(summary["by_type"]["answerable"]["refusal_correct_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
