"""Optional LangGraph orchestration for the knowledge Agent.

The hand-written :class:`~rag_agent.agent.runtime.KnowledgeAgent` remains the
default and requires no graph dependency.  This module reuses the same chat
provider, tool validation, history contract, and ``AgentState`` shape while
exposing three observable nodes: ``agent`` (model decision), ``tools``
(read-only tool execution), and ``finalize`` (history update).  LangGraph is
imported only when the graph is built, so offline ingestion/search still work
when the optional package is absent.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from typing import Any, Mapping, TypedDict

from .history import ConversationHistory
from .knowledge_tool import SEARCH_KNOWLEDGE_TOOL
from .runtime import AgentResult, AgentState, KnowledgeAgent


GRAPH_AGENT_NODE = "agent"
GRAPH_TOOLS_NODE = "tools"
GRAPH_FINALIZE_NODE = "finalize"
MAX_STEP_ANSWER = "Agent 达到最大工具调用轮数，已停止以避免无限请求。"


class GraphUnavailableError(RuntimeError):
    """Raised when the optional LangGraph dependency is not installed."""


class GraphState(TypedDict, total=False):
    """Plain-data state exchanged by LangGraph nodes.

    Keeping history as its JSON dictionary form makes the graph state suitable
    for LangGraph checkpoints and makes ``graph.invoke(...).to_dict`` easy to
    audit.  ``pending_tool_calls`` is transient and is cleared before finalize.
    """

    question: str
    messages: list[dict[str, Any]]
    pending_tool_calls: list[dict[str, str]]
    tool_calls: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    step: int
    answer: str
    stopped_reason: str
    history: dict[str, Any]


def langgraph_available() -> bool:
    """Return whether importing the optional ``langgraph`` package is possible."""

    try:
        return importlib.util.find_spec("langgraph") is not None
    except (ImportError, ValueError):
        return False


def initial_graph_state(
    agent: KnowledgeAgent,
    question: str,
    *,
    history: ConversationHistory | None = None,
) -> GraphState:
    """Build a validated graph input from the shared Agent history contract."""

    clean_question = question.strip() if isinstance(question, str) else ""
    if not clean_question:
        raise ValueError("问题不能为空")
    conversation = history if history is not None else ConversationHistory()
    if not isinstance(conversation, ConversationHistory):
        raise TypeError("history 必须是 ConversationHistory")
    return {
        "question": clean_question,
        "messages": [
            {"role": "system", "content": agent._system_prompt},
            *conversation.to_messages(),
            {"role": "user", "content": clean_question},
        ],
        "pending_tool_calls": [],
        "tool_calls": [],
        "evidence": [],
        "step": 0,
        "answer": "",
        "stopped_reason": "",
        "history": conversation.to_dict(),
    }


def agent_model_node(agent: KnowledgeAgent, state: Mapping[str, Any]) -> GraphState:
    """Ask the model for an answer or structured search calls."""

    messages = _messages(state)
    step = _step(state)
    if step >= agent.max_steps:
        return {
            "answer": MAX_STEP_ANSWER,
            "stopped_reason": "max_steps",
            "pending_tool_calls": [],
            "messages": messages,
            "step": step,
        }

    turn = agent.chat_provider.complete_with_tools(messages, [SEARCH_KNOWLEDGE_TOOL])
    messages.append(deepcopy(turn.assistant_message))
    next_step = step + 1
    pending = [
        {
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments,
        }
        for call in turn.tool_calls
    ]
    if pending:
        return {
            "messages": messages,
            "pending_tool_calls": pending,
            "step": next_step,
            "answer": "",
            "stopped_reason": "",
        }
    answer = (turn.content or "").strip() or "聊天模型没有返回可用答案。"
    return {
        "messages": messages,
        "pending_tool_calls": [],
        "step": next_step,
        "answer": answer,
        "stopped_reason": "completed",
    }


def agent_tools_node(agent: KnowledgeAgent, state: Mapping[str, Any]) -> GraphState:
    """Execute pending calls through the same read-only tool boundary."""

    messages = _messages(state)
    calls_audit = _dict_list(state.get("tool_calls"))
    evidence = _dict_list(state.get("evidence"))
    step = _step(state)

    for raw_call in state.get("pending_tool_calls", []) or []:
        call = raw_call if isinstance(raw_call, Mapping) else {}
        call_id = call.get("call_id", "")
        name = call.get("name", "")
        arguments = call.get("arguments", "")
        call_id = call_id if isinstance(call_id, str) else str(call_id)
        name = name if isinstance(name, str) else str(name)
        audit: dict[str, Any] = {
            "step": step,
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }
        if name != SEARCH_KNOWLEDGE_TOOL["name"]:
            output: dict[str, Any] = {"error": f"不允许的工具：{name}"}
        elif not isinstance(arguments, str):
            output = {"error": "工具参数必须是 JSON 字符串"}
        else:
            try:
                output_text = agent.tool.invoke_json(arguments)
                output_value = json.loads(output_text)
                output = output_value if isinstance(output_value, dict) else {"error": "工具返回格式无效"}
            except Exception as error:  # Validation/provider errors remain auditable evidence.
                output = {"error": str(error)}
        audit["result"] = output
        calls_audit.append(audit)
        for result in output.get("results", []) or []:
            if isinstance(result, dict):
                evidence.append(deepcopy(result))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(output, ensure_ascii=False),
            }
        )

    stopped_reason = "max_steps" if step >= agent.max_steps else ""
    return {
        "messages": messages,
        "pending_tool_calls": [],
        "tool_calls": calls_audit,
        "evidence": evidence,
        "answer": MAX_STEP_ANSWER if stopped_reason else "",
        "stopped_reason": stopped_reason,
        "step": step,
    }


def finalize_graph_node(state: Mapping[str, Any]) -> GraphState:
    """Append only completed answers to history and clear transient calls."""

    history = _history(state.get("history"))
    if state.get("stopped_reason") == "completed":
        history = history.append(
            _required_text(state.get("question"), "question"),
            _required_text(state.get("answer"), "answer"),
        )
    return {
        "history": history.to_dict(),
        "pending_tool_calls": [],
    }


def route_after_agent(state: Mapping[str, Any]) -> str:
    """Route model output to tools or finalization."""

    return GRAPH_TOOLS_NODE if state.get("pending_tool_calls") else GRAPH_FINALIZE_NODE


def route_after_tools(agent: KnowledgeAgent, state: Mapping[str, Any]) -> str:
    """Stop after the configured model turns, otherwise ask the model again."""

    if state.get("stopped_reason") or _step(state) >= agent.max_steps:
        return GRAPH_FINALIZE_NODE
    return GRAPH_AGENT_NODE


def build_knowledge_graph(agent: KnowledgeAgent) -> Any:
    """Compile the optional LangGraph graph for one configured Agent.

    Importing this module remains safe without LangGraph; only this function
    requires the extra package (``python -m pip install '.[graph]'``).
    """

    try:
        from langgraph.graph import StateGraph
    except (ImportError, ModuleNotFoundError) as error:
        raise GraphUnavailableError(
            "LangGraph 未安装；如需 --graph，请安装可选依赖："
            "python -m pip install '.[graph]'"
        ) from error

    workflow = StateGraph(GraphState)
    workflow.add_node(GRAPH_AGENT_NODE, lambda state: agent_model_node(agent, state))
    workflow.add_node(GRAPH_TOOLS_NODE, lambda state: agent_tools_node(agent, state))
    workflow.add_node(GRAPH_FINALIZE_NODE, finalize_graph_node)
    workflow.set_entry_point(GRAPH_AGENT_NODE)
    workflow.add_conditional_edges(
        GRAPH_AGENT_NODE,
        route_after_agent,
        {
            GRAPH_TOOLS_NODE: GRAPH_TOOLS_NODE,
            GRAPH_FINALIZE_NODE: GRAPH_FINALIZE_NODE,
        },
    )
    workflow.add_conditional_edges(
        GRAPH_TOOLS_NODE,
        lambda state: route_after_tools(agent, state),
        {
            GRAPH_AGENT_NODE: GRAPH_AGENT_NODE,
            GRAPH_FINALIZE_NODE: GRAPH_FINALIZE_NODE,
        },
    )
    workflow.set_finish_point(GRAPH_FINALIZE_NODE)
    return workflow.compile()


def run_knowledge_graph(
    agent: KnowledgeAgent,
    question: str,
    *,
    history: ConversationHistory | None = None,
    config: Mapping[str, Any] | None = None,
) -> AgentResult:
    """Invoke a compiled graph and convert its final state to ``AgentResult``."""

    graph = build_knowledge_graph(agent)
    initial = initial_graph_state(agent, question, history=history)
    if config is None:
        result = graph.invoke(initial)
    else:
        result = graph.invoke(initial, config=dict(config))
    if not isinstance(result, Mapping):
        raise GraphUnavailableError("LangGraph 返回了无法解析的状态")
    return agent_result_from_graph_state(agent, result)


def agent_result_from_graph_state(
    agent: KnowledgeAgent,
    state: Mapping[str, Any],
) -> AgentResult:
    """Convert a graph checkpoint/result to the shared public result contract."""

    history = _history(state.get("history"))
    question = _required_text(state.get("question"), "question")
    answer = _required_text(state.get("answer"), "answer")
    stopped_reason = _required_text(state.get("stopped_reason"), "stopped_reason")
    messages = tuple(deepcopy(_messages(state)))
    tool_calls = tuple(deepcopy(_dict_list(state.get("tool_calls"))))
    evidence = tuple(deepcopy(_dict_list(state.get("evidence"))))
    snapshot = AgentState(
        question=question,
        messages=messages,
        tool_calls=tool_calls,
        evidence=evidence,
        step=_step(state),
        answer=answer,
        stopped_reason=stopped_reason,
        history=history,
    )
    return AgentResult(
        question=question,
        answer=answer,
        tool_calls=tool_calls,
        evidence=evidence,
        used_model=agent.chat_provider.model,
        stopped_reason=stopped_reason,
        history=history,
        state=snapshot,
    )


def _messages(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = state.get("messages", [])
    if not isinstance(raw, list):
        raise ValueError("图状态 messages 必须是数组")
    if any(not isinstance(message, dict) for message in raw):
        raise ValueError("图状态 messages 必须是对象数组")
    return deepcopy(raw)


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("图状态审计字段必须是数组")
    if any(not isinstance(item, dict) for item in value):
        raise ValueError("图状态审计字段必须是对象数组")
    return deepcopy(value)


def _step(state: Mapping[str, Any]) -> int:
    value = state.get("step", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("图状态 step 必须是非负整数")
    return value


def _history(value: Any) -> ConversationHistory:
    if isinstance(value, ConversationHistory):
        return value
    if isinstance(value, Mapping):
        return ConversationHistory.from_dict(value)
    raise ValueError("图状态 history 必须是历史快照对象")


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"图状态 {field_name} 必须是非空字符串")
    return value.strip()
