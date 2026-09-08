"""Agent 评测 Harness：题库、确定性指标、裁判与跑批编排。

设计文档见 ``docs/Agent评测Harness工程文档.md``。双模式原则：fake 模式
（免费，理想剧本假模型应得满分）验证尺子自身与机制回归；real 模式测
回答质量。两种结论不混用。
"""

from .tasks import (
    TASKS_SCHEMA_VERSION,
    HarnessTask,
    load_tasks,
)
from .metrics import (
    CitationStatus,
    DeterministicMetrics,
    aggregate_metrics,
    evaluate_task,
    refusal_detected,
)
from .runner import (
    HarnessOptions,
    IdealScriptChat,
    compare_reports,
    run_harness,
)

__all__ = [
    "TASKS_SCHEMA_VERSION",
    "HarnessTask",
    "load_tasks",
    "CitationStatus",
    "DeterministicMetrics",
    "evaluate_task",
    "aggregate_metrics",
    "refusal_detected",
    "HarnessOptions",
    "IdealScriptChat",
    "run_harness",
    "compare_reports",
]
