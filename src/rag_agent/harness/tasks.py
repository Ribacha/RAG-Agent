"""评测题库：加载、校验与数据契约（schema v1）。

格式沿用 ``evaluation.py`` 的标注约定（relevant_chunk_ids /
relevant_source_paths 命中任一即算命中）并扩展：task_type 区分应有答案题
与应拒答的陷阱题，expected_points 是裁判覆盖度评分的依据。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..storage.jsonl import read_jsonl


TASKS_SCHEMA_VERSION = 1
TASK_TYPES = ("answerable", "refusal")
MAX_EXPECTED_POINTS = 8


@dataclass(frozen=True)
class HarnessTask:
    """一道评测题及其标注。"""

    task_id: str
    query: str
    task_type: str
    relevant_chunk_ids: tuple[str, ...] = ()
    relevant_source_paths: tuple[str, ...] = ()
    expected_points: tuple[str, ...] = ()
    category: str = ""
    auto: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "query": self.query,
            "task_type": self.task_type,
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "relevant_source_paths": list(self.relevant_source_paths),
            "expected_points": list(self.expected_points),
            "category": self.category,
            "auto": self.auto,
        }

    @classmethod
    def from_dict(cls, row: Any, *, line_number: int) -> "HarnessTask":
        """解析并逐项校验一行题库；错误信息带行号便于定位。"""

        prefix = f"题库第 {line_number} 行"
        if not isinstance(row, dict):
            raise ValueError(f"{prefix}必须是 JSON 对象")

        task_id = _required_text(row.get("id"), f"{prefix} id")
        query = _required_text(row.get("query"), f"{prefix} query")
        task_type = row.get("task_type", "answerable")
        if task_type not in TASK_TYPES:
            raise ValueError(
                f"{prefix} task_type 必须是 {'/'.join(TASK_TYPES)}：{task_type!r}"
            )

        chunk_ids = _label_list(row.get("relevant_chunk_ids"), "relevant_chunk_ids", prefix)
        source_paths = _label_list(
            row.get("relevant_source_paths"), "relevant_source_paths", prefix
        )
        if task_type == "answerable" and not chunk_ids and not source_paths:
            raise ValueError(
                f"{prefix} answerable 题至少需要一种相关标注"
                "（relevant_chunk_ids 或 relevant_source_paths）"
            )

        raw_points = row.get("expected_points", [])
        if not isinstance(raw_points, list):
            raise ValueError(f"{prefix} expected_points 必须是数组")
        points = tuple(
            _required_text(item, f"{prefix} expected_points[{index}]")
            for index, item in enumerate(raw_points)
        )
        if len(points) > MAX_EXPECTED_POINTS:
            raise ValueError(
                f"{prefix} expected_points 最多 {MAX_EXPECTED_POINTS} 条"
            )
        if task_type == "answerable" and not points:
            raise ValueError(
                f"{prefix} answerable 题需要 1-{MAX_EXPECTED_POINTS} 条 expected_points"
                "（裁判覆盖度评分的依据）"
            )

        category = row.get("category", "")
        if not isinstance(category, str):
            raise ValueError(f"{prefix} category 必须是字符串")
        auto = row.get("auto", False)
        if not isinstance(auto, bool):
            raise ValueError(f"{prefix} auto 必须是布尔值")

        return cls(
            task_id=task_id,
            query=query,
            task_type=task_type,
            relevant_chunk_ids=chunk_ids,
            relevant_source_paths=source_paths,
            expected_points=points,
            category=category.strip(),
            auto=auto,
        )


def load_tasks(path: Path) -> list[HarnessTask]:
    """加载题库 JSONL；id 唯一性在整库级别校验。"""

    rows = list(read_jsonl(path))
    tasks: list[HarnessTask] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        task = HarnessTask.from_dict(row, line_number=line_number)
        if task.task_id in seen_ids:
            raise ValueError(f"题库 id 重复：{task.task_id}（第 {line_number} 行）")
        seen_ids.add(task.task_id)
        tasks.append(task)
    return tasks


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} 必须是字符串")
    clean = value.strip()
    if not clean:
        raise ValueError(f"{label} 不能为空")
    return clean


def _label_list(value: Any, field_name: str, prefix: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{prefix} {field_name} 必须是数组")
    labels = tuple(
        _required_text(item, f"{prefix} {field_name}[{index}]")
        for index, item in enumerate(value)
    )
    return labels
