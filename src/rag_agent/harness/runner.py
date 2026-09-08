"""跑批编排：题库 -> 逐题执行 agent -> 指标 -> 聚合 -> 报告落盘与对比。

fake 模式注入"理想剧本"假模型（见 IdealScriptChat），零成本、确定性，
用于验证 harness 自身与工作流机制；real 模式由调用方传入真实聊天模型
工厂。两种模式的结论不混用（工程文档第 7 节）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any, Callable

from ..agent import KnowledgeAgent, KnowledgeSearchTool
from ..answering.chat import ToolCall, ToolChatTurn
from ..answering.rag import DEFAULT_NO_EVIDENCE
from ..embeddings import create_embedding_provider
from ..models import sha256_bytes
from ..retrieval.index import LocalVectorIndex
from ..workspace import find_workspace_root
from .metrics import DeterministicMetrics, aggregate_metrics, evaluate_task
from .tasks import HarnessTask, load_tasks


HARNESS_REPORT_SCHEMA_VERSION = 1
# 理想剧本的作答门槛：证据最高分低于此值视为"知识库没有"，按陷阱题处理。
IDEAL_ANSWER_THRESHOLD = 0.15


class IdealScriptChat:
    """理想剧本假模型（fake 模式）：机械化执行工作流。

    策略：先检索（任务原句）；证据非空且最高分 >= answer_threshold 时
    引用 [1] 作答，否则输出标准拒答文案；永不触网。它验证的是尺子自身
    与工作流机制，**不代表真实模型质量**。
    """

    model = "ideal-script"

    def __init__(self, answer_threshold: float = IDEAL_ANSWER_THRESHOLD) -> None:
        self.answer_threshold = answer_threshold

    def complete_with_tools(self, messages, tools):
        last_user = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        query = str((last_user or {}).get("content", "")).strip() or "问题"
        last_tool = next(
            (m for m in reversed(messages) if m.get("role") == "tool"), None
        )
        if last_tool is None:
            call = ToolCall(
                "ideal-search",
                "search_knowledge_base",
                json.dumps({"query": query, "top_k": 5}, ensure_ascii=False),
            )
            return ToolChatTurn(
                None,
                (call,),
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments,
                            },
                        }
                    ],
                },
            )
        try:
            payload = json.loads(str(last_tool.get("content", "{}")))
            results = [r for r in payload.get("results", []) or [] if isinstance(r, dict)]
        except json.JSONDecodeError:
            results = []
        top_score = max(
            (float(r.get("score", 0.0)) for r in results), default=0.0
        )
        if results and top_score >= self.answer_threshold:
            answer = "基于检索证据，这是对问题的回答 [1]。"
            return ToolChatTurn(answer, (), {"role": "assistant", "content": answer})
        return ToolChatTurn(
            DEFAULT_NO_EVIDENCE, (), {"role": "assistant", "content": DEFAULT_NO_EVIDENCE}
        )


@dataclass(frozen=True)
class HarnessOptions:
    """一次 harness 运行的配置。"""

    tasks_path: Path
    index_path: Path
    mode: str = "fake"  # fake | real
    repeat: int = 1
    limit: int | None = None
    report_dir: Path | None = None
    compare_path: Path | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("fake", "real"):
            raise ValueError("mode 必须是 fake 或 real")
        if self.repeat < 1:
            raise ValueError("repeat 必须 >= 1")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit 必须 >= 1")


def run_harness(
    options: HarnessOptions,
    *,
    chat_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """执行一次评测并返回完整报告（同时落盘）。"""

    tasks = load_tasks(options.tasks_path)
    if options.limit is not None:
        tasks = tasks[: options.limit]
    if not tasks:
        raise ValueError("题库为空")

    index = LocalVectorIndex.load(options.index_path)
    provider = create_embedding_provider("hash", dimension=index.dimension)
    search_tool = KnowledgeSearchTool(index, provider)

    if options.mode == "real":
        if chat_factory is None:
            raise ValueError("real 模式需要提供 chat_factory")
        chat = chat_factory()
        web_tool_factory: Callable[[], Any] = lambda: _build_web_tool()
        model_name = str(getattr(chat, "model", "real"))
    else:
        # fake 模式强制理想剧本并禁网：结论只关于尺子与机制。
        chat = IdealScriptChat()
        web_tool_factory = lambda: None
        model_name = IdealScriptChat.model

    per_run_rows: list[list[tuple[HarnessTask, DeterministicMetrics, dict]]] = []
    for _run in range(options.repeat):
        web_tool = web_tool_factory()
        rows: list[tuple[HarnessTask, DeterministicMetrics, dict]] = []
        for task in tasks:
            agent = KnowledgeAgent(
                search_tool,
                chat,
                max_steps=8 if web_tool is not None else 5,
                web_tool=web_tool,
            )
            result = agent.run(task.query)
            payload = result.to_dict()
            payload["web_tool_enabled"] = web_tool is not None
            rows.append((task, evaluate_task(task, payload), payload))
        per_run_rows.append(rows)

    final_rows = per_run_rows[-1]
    report: dict[str, Any] = {
        "schema_version": HARNESS_REPORT_SCHEMA_VERSION,
        "mode": options.mode,
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_count": len(tasks),
        "repeat": options.repeat,
        "tasks_fingerprint": sha256_bytes(options.tasks_path.read_bytes()),
        "aggregate": aggregate_metrics([(t, m) for t, m, _ in final_rows]),
        "jitter": None,
        "judge_calibration": None,
        "tasks": [
            {
                "task": task.to_dict(),
                "metrics": metric.to_dict(),
                "result": {
                    "answer": payload.get("answer", ""),
                    "evidence": payload.get("evidence", []),
                    "tool_calls": payload.get("tool_calls", []),
                    "stopped_reason": payload.get("stopped_reason", ""),
                    "web_tool_enabled": payload.get("web_tool_enabled", False),
                },
                "judged": None,
            }
            for task, metric, payload in final_rows
        ],
    }
    if options.repeat > 1:
        report["jitter"] = _jitter_section(per_run_rows, tasks)
    report_path = _write_report(report, options.report_dir)
    report["report_path"] = str(report_path)
    if options.compare_path is not None:
        previous = json.loads(options.compare_path.read_text(encoding="utf-8"))
        report["compare"] = compare_reports(previous, report)
    return report


def compare_reports(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """对比两份报告的关键指标与逐题拒答翻转。"""

    old_overall = old.get("aggregate", {}).get("overall", {})
    new_overall = new.get("aggregate", {}).get("overall", {})
    keys = (
        "retrieval_hit_rate",
        "refusal_correct_rate",
        "citation_ok_rate",
        "avg_tool_calls",
        "web_used_count",
    )
    deltas = {
        key: {
            "old": old_overall.get(key),
            "new": new_overall.get(key),
        }
        for key in keys
        if old_overall.get(key) is not None or new_overall.get(key) is not None
    }
    old_refused = {
        row.get("task", {}).get("id"): row.get("metrics", {}).get("refused")
        for row in old.get("tasks", [])
    }
    new_refused = {
        row.get("task", {}).get("id"): row.get("metrics", {}).get("refused")
        for row in new.get("tasks", [])
    }
    flips = [
        task_id
        for task_id, refused in new_refused.items()
        if task_id in old_refused and old_refused[task_id] != refused
    ]
    old_judge = old.get("judge_summary") or {}
    new_judge = new.get("judge_summary") or {}
    judge_delta = (
        {
            "old_overall": old_judge.get("overall_mean"),
            "new_overall": new_judge.get("overall_mean"),
        }
        if old_judge or new_judge
        else None
    )
    return {
        "old_created_at": old.get("created_at"),
        "old_model": old.get("model"),
        "metric_deltas": deltas,
        "refusal_flips_between_runs": flips,
        "judge_delta": judge_delta,
    }


def _jitter_section(
    per_run_rows: list[list[tuple[HarnessTask, DeterministicMetrics, dict]]],
    tasks: list[HarnessTask],
) -> dict[str, Any]:
    run_summaries = [
        aggregate_metrics([(t, m) for t, m, _ in rows])["overall"]
        for rows in per_run_rows
    ]

    def mean_std(key: str) -> dict[str, Any] | None:
        values = [
            summary.get(key)
            for summary in run_summaries
            if isinstance(summary.get(key), (int, float))
        ]
        if not values:
            return None
        return {
            "mean": round(statistics.mean(values), 4),
            "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        }

    refused_by_task: dict[str, list[bool]] = {task.task_id: [] for task in tasks}
    for rows in per_run_rows:
        for task, metric, _ in rows:
            refused_by_task[task.task_id].append(metric.refused)
    flips = [
        {"task_id": task_id, "states": states}
        for task_id, states in refused_by_task.items()
        if len(set(states)) > 1
    ]
    return {
        "run_count": len(per_run_rows),
        "refusal_correct_rate": mean_std("refusal_correct_rate"),
        "citation_ok_rate": mean_std("citation_ok_rate"),
        "retrieval_hit_rate": mean_std("retrieval_hit_rate"),
        "refusal_flip_count": len(flips),
        "refusal_flips": flips,
    }


def _write_report(report: dict[str, Any], report_dir: Path | None) -> Path:
    directory = report_dir or find_workspace_root() / "data/eval/reports"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = directory / f"harness-{stamp}.json"
    counter = 1
    while path.exists():  # 同秒多次运行时避免覆盖
        path = directory / f"harness-{stamp}-{counter}.json"
        counter += 1
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def _build_web_tool():
    from ..agent import WebFetchTool

    return WebFetchTool()
