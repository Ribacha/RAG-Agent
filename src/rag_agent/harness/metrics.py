"""确定性指标：零成本的每题判定（检索命中/引用真实性/拒答正确性/效率）。

输入是 ``AgentResult.to_dict()`` 的审计载荷，纯函数、完全离线；
指标定义见工程文档 4.1 节。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from ..answering.rag import DEFAULT_NO_EVIDENCE
from .tasks import HarnessTask


CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class CitationStatus:
    """引用真实性判定。"""

    OK = "ok"
    INVALID = "invalid"  # 引用了不存在的编号
    UNCITED = "uncited"  # 有实质结论但一个引用都没有
    NOT_APPLICABLE = "not_applicable"  # 拒答题无需引用


@dataclass(frozen=True)
class DeterministicMetrics:
    """一道题的确定性指标。"""

    task_id: str
    retrieval_hit: bool | None  # 无标注（refusal 题）时为 None
    citation_status: str
    refused: bool
    refusal_correct: bool
    tool_calls: int
    web_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "retrieval_hit": self.retrieval_hit,
            "citation_status": self.citation_status,
            "refused": self.refused,
            "refusal_correct": self.refusal_correct,
            "tool_calls": self.tool_calls,
            "web_used": self.web_used,
        }


def refusal_detected(answer: str, evidence_count: int) -> bool:
    """判定一次回答是否属于拒答。

    两种形态：标准拒答文案；或证据为空且回答明确说"没有找到"类内容
    （模型自己的拒答措辞）。
    """

    clean = (answer or "").strip()
    if clean == DEFAULT_NO_EVIDENCE.strip():
        return True
    if evidence_count == 0 and any(
        marker in clean for marker in ("没有找到", "无相关", "无法找到", "未找到")
    ):
        return True
    return False


def evaluate_task(task: HarnessTask, result: Mapping[str, Any]) -> DeterministicMetrics:
    """从 AgentResult 的 to_dict() 载荷计算确定性指标。"""

    answer = str(result.get("answer", ""))
    evidence = [item for item in result.get("evidence", []) or [] if isinstance(item, dict)]
    tool_calls = [call for call in result.get("tool_calls", []) or [] if isinstance(call, dict)]

    refused = refusal_detected(answer, len(evidence))
    citation_status = _citation_status(answer, evidence_count=len(evidence), refused=refused)
    retrieval_hit = (
        _retrieval_hit(task, tool_calls) if task.relevant_chunk_ids or task.relevant_source_paths else None
    )
    # 拒答正确性：refusal 题应该拒；answerable 题不应该拒。
    refusal_correct = refused if task.task_type == "refusal" else not refused

    return DeterministicMetrics(
        task_id=task.task_id,
        retrieval_hit=retrieval_hit,
        citation_status=citation_status,
        refused=refused,
        refusal_correct=refusal_correct,
        tool_calls=len(tool_calls),
        web_used=any(call.get("name") == "fetch_web_page" for call in tool_calls),
    )


def _citation_status(answer: str, *, evidence_count: int, refused: bool) -> str:
    if refused or evidence_count == 0:
        # 没有证据就没有可引用的对象；拒答是正确行为而非缺陷。
        return CitationStatus.NOT_APPLICABLE
    numbers = [
        int(match.group(1))
        for match in CITATION_PATTERN.finditer(answer)
    ]
    if not numbers:
        # 有证据、有实质结论，却一个 [n] 都没有 —— 需要人工/裁判关注的降级信号。
        return CitationStatus.UNCITED
    if all(1 <= number <= evidence_count for number in numbers):
        return CitationStatus.OK
    return CitationStatus.INVALID


def _retrieval_hit(task: HarnessTask, tool_calls: Sequence[Mapping[str, Any]]) -> bool:
    """扫审计里的 search 结果，命中任一标注即算命中（与 evaluation.py 口径一致）。"""

    labeled_chunks = set(task.relevant_chunk_ids)
    labeled_sources = set(task.relevant_source_paths)
    for call in tool_calls:
        if call.get("name") != "search_knowledge_base":
            continue
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        for item in result.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("chunk_id", "")) in labeled_chunks:
                return True
            if str(item.get("source_path", "")) in labeled_sources:
                return True
    return False


def aggregate_metrics(
    rows: Sequence[tuple[HarnessTask, DeterministicMetrics]],
) -> dict[str, Any]:
    """聚合逐题指标；answerable/refusal 与 auto/人工分开统计。"""

    def _summary(tasks: list[HarnessTask], metrics: list[DeterministicMetrics]) -> dict[str, Any]:
        if not metrics:
            return {"count": 0}
        answerable_hit = [
            m.retrieval_hit for t, m in zip(tasks, metrics) if t.task_type == "answerable"
        ]
        citations = [m.citation_status for m in metrics]
        return {
            "count": len(metrics),
            "retrieval_hit_rate": (
                round(sum(1 for hit in answerable_hit if hit) / len(answerable_hit), 4)
                if answerable_hit
                else None
            ),
            "refusal_correct_rate": round(
                sum(1 for m in metrics if m.refusal_correct) / len(metrics), 4
            ),
            "citation_ok_rate": round(
                sum(1 for status in citations if status == CitationStatus.OK) / len(citations), 4
            ),
            "citation_uncited": sum(1 for status in citations if status == CitationStatus.UNCITED),
            "citation_invalid": sum(1 for status in citations if status == CitationStatus.INVALID),
            "avg_tool_calls": round(sum(m.tool_calls for m in metrics) / len(metrics), 2),
            "web_used_count": sum(1 for m in metrics if m.web_used),
        }

    tasks = [task for task, _ in rows]
    metrics = [metric for _, metric in rows]
    return {
        "overall": _summary(tasks, metrics),
        "by_type": {
            task_type: _summary(
                [t for t, m in rows if t.task_type == task_type],
                [m for t, m in rows if t.task_type == task_type],
            )
            for task_type in ("answerable", "refusal")
        },
        "by_provenance": {
            label: _summary(
                [t for t, m in rows if t.auto == (label == "auto")],
                [m for t, m in rows if t.auto == (label == "auto")],
            )
            for label in ("auto", "human")
        },
        "by_category": {
            category: _summary(
                [t for t, m in rows if t.category == category],
                [m for t, m in rows if t.category == category],
            )
            for category in sorted({t.category for t in tasks if t.category})
        },
    }
