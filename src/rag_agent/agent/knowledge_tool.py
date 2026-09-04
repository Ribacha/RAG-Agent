"""The single read-only tool exposed to a future chat Agent.

Keeping this boundary separate from the model client is intentional.  The
model can request a search with structured arguments, while Python validates
those arguments and performs the actual index lookup.  It never receives an
arbitrary path or a function that can modify source files.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from ..embeddings.base import EmbeddingProvider
from ..retrieval.index import LocalVectorIndex


class KnowledgeToolError(ValueError):
    """Raised when a model-supplied search request is invalid."""


SEARCH_KNOWLEDGE_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "search_knowledge_base",
    "description": (
        "在已经构建的知识库中查找与问题最相关的证据。"
        "只能用于检索，不能读取任意文件、执行命令或修改资料。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要检索的自然语言问题或关键词",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
                "description": "返回的证据片段数量，最多 20 条",
            },
            "min_score": {
                "type": "number",
                "minimum": -1,
                "maximum": 1,
                "default": 0.08,
                "description": "最低余弦相似度阈值",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass(frozen=True)
class KnowledgeSearchTool:
    """Read-only adapter over :class:`LocalVectorIndex`."""

    index: LocalVectorIndex
    embedding_provider: EmbeddingProvider
    default_top_k: int = 5
    default_min_score: float = 0.08

    def __post_init__(self) -> None:
        if not 1 <= self.default_top_k <= 20:
            raise ValueError("default_top_k 必须在 1 到 20 之间")
        if not -1 <= self.default_min_score <= 1:
            raise ValueError("default_min_score 必须在 -1 到 1 之间")

    def invoke(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> dict[str, Any]:
        """Validate arguments, run search, and return JSON-safe evidence."""

        clean_query = _validate_query(query)
        resolved_top_k = self.default_top_k if top_k is None else _validate_top_k(top_k)
        resolved_min_score = (
            self.default_min_score
            if min_score is None
            else _validate_min_score(min_score)
        )
        results = self.index.search(
            clean_query,
            provider=self.embedding_provider,
            top_k=resolved_top_k,
            min_score=resolved_min_score,
        )
        return {
            "query": clean_query,
            "top_k": resolved_top_k,
            "min_score": resolved_min_score,
            "count": len(results),
            "results": [result.to_dict() for result in results],
        }

    def invoke_json(self, arguments_json: str) -> str:
        """Parse a model function-call payload and return structured JSON."""

        try:
            arguments = json.loads(arguments_json)
        except json.JSONDecodeError as error:
            raise KnowledgeToolError(f"工具参数不是有效 JSON：{error.msg}") from error
        if not isinstance(arguments, dict):
            raise KnowledgeToolError("工具参数必须是 JSON 对象")
        unknown = set(arguments) - {"query", "top_k", "min_score"}
        if unknown:
            raise KnowledgeToolError(
                "工具参数包含未声明字段：" + ", ".join(sorted(unknown))
            )
        try:
            result = self.invoke(
                arguments.get("query"),
                top_k=arguments.get("top_k"),
                min_score=arguments.get("min_score"),
            )
        except (TypeError, ValueError) as error:
            raise KnowledgeToolError(str(error)) from error
        return json.dumps(result, ensure_ascii=False, sort_keys=True)


def _validate_query(query: Any) -> str:
    if not isinstance(query, str):
        raise KnowledgeToolError("query 必须是字符串")
    clean = query.strip()
    if not clean:
        raise KnowledgeToolError("query 不能为空")
    if len(clean) > 4000:
        raise KnowledgeToolError("query 不能超过 4000 个字符")
    return clean


def _validate_top_k(value: Any) -> int:
    # bool is an int subclass but is not a meaningful tool argument.
    if isinstance(value, bool) or not isinstance(value, int):
        raise KnowledgeToolError("top_k 必须是整数")
    if not 1 <= value <= 20:
        raise KnowledgeToolError("top_k 必须在 1 到 20 之间")
    return value


def _validate_min_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise KnowledgeToolError("min_score 必须是数字")
    numeric = float(value)
    if not -1 <= numeric <= 1:
        raise KnowledgeToolError("min_score 必须在 -1 到 1 之间")
    return numeric
