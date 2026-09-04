"""Optional OpenAI-compatible chat client.

The rest of the project does not import the SDK.  This keeps ingestion and
offline search usable on a machine without network access or an API key.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class ToolCall:
    """Normalized function call returned by a chat provider."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolChatTurn:
    """Assistant message plus normalized tool calls for the next loop step."""

    content: str | None
    tool_calls: tuple[ToolCall, ...]
    assistant_message: dict[str, Any]


class ChatError(RuntimeError):
    """Raised when a chat provider cannot produce an answer."""


class ChatProvider(Protocol):
    """Minimal interface consumed by :class:`RagAnswerer`."""

    @property
    def model(self) -> str:
        """Configured chat model name."""

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        """Return the assistant's text for a chat message sequence."""


class ToolCallingChatProvider(Protocol):
    """Chat provider capability needed by the tool-using Agent loop."""

    @property
    def model(self) -> str:
        """Configured chat model name."""

    def complete_with_tools(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ToolChatTurn:
        """Return an assistant turn which may contain function calls."""


@dataclass
class OpenAICompatibleChatProvider:
    """Chat Completions client for OpenAI-compatible services such as DeepSeek."""

    api_key: str
    model: str
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = 1200
    _client: Any | None = None

    @classmethod
    def from_environment(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = 1200,
    ) -> "OpenAICompatibleChatProvider":
        key = api_key or os.getenv("LLM_API_KEY") or os.getenv("CHAT_API_KEY")
        if not key:
            raise ChatError(
                "缺少聊天模型 API Key；请设置 LLM_API_KEY/CHAT_API_KEY，"
                "或通过 --llm-api-key 传入"
            )
        return cls(
            api_key=key,
            base_url=base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("CHAT_BASE_URL")
            or "https://api.deepseek.com",
            model=model
            or os.getenv("LLM_MODEL")
            or os.getenv("CHAT_MODEL")
            or "deepseek-chat",
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        if not messages:
            raise ChatError("消息列表不能为空")
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        try:
            response = client.chat.completions.create(**kwargs)
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise ChatError("聊天接口没有返回 choices")
            content = getattr(choices[0].message, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
            # A few compatible APIs return a list of typed content parts.
            if isinstance(content, list):
                parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                ]
                text = "".join(parts).strip()
                if text:
                    return text
            raise ChatError("聊天接口返回了空内容")
        except ChatError:
            raise
        except Exception as error:  # SDK/provider exception types vary.
            raise ChatError(f"聊天模型请求失败：{error}") from error

    def complete_with_tools(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ToolChatTurn:
        """Call Chat Completions with tools and normalize its response."""

        if not messages:
            raise ChatError("消息列表不能为空")
        if not tools:
            raise ChatError("工具列表不能为空")
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "tools": [_as_chat_tool_schema(tool) for tool in tools],
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        try:
            response = client.chat.completions.create(**kwargs)
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise ChatError("聊天接口没有返回 choices")
            message = choices[0].message
            content = getattr(message, "content", None)
            if not isinstance(content, str):
                content = None
            calls: list[ToolCall] = []
            serialized_calls: list[dict[str, Any]] = []
            for item in getattr(message, "tool_calls", None) or []:
                function = getattr(item, "function", None)
                name = getattr(function, "name", None)
                arguments = getattr(function, "arguments", None)
                call_id = getattr(item, "id", None)
                if not all(isinstance(value, str) for value in (name, arguments, call_id)):
                    raise ChatError("聊天接口返回了格式不完整的 tool_call")
                calls.append(ToolCall(call_id=call_id, name=name, arguments=arguments))
                serialized_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content,
            }
            if serialized_calls:
                assistant_message["tool_calls"] = serialized_calls
            return ToolChatTurn(
                content=content,
                tool_calls=tuple(calls),
                assistant_message=assistant_message,
            )
        except ChatError:
            raise
        except Exception as error:  # SDK/provider exception types vary.
            raise ChatError(f"聊天模型工具调用请求失败：{error}") from error

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ModuleNotFoundError as error:
            raise ChatError(
                "聊天模型需要 OpenAI SDK，请安装 `python -m pip install '.[llm]'`"
            ) from error
        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client


def _as_chat_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    """Accept Responses-style flat schemas and emit Chat Completions shape."""

    if "function" in tool:
        return tool
    name = tool.get("name")
    parameters = tool.get("parameters")
    if not isinstance(name, str) or not isinstance(parameters, dict):
        raise ChatError("工具 schema 缺少 name 或 parameters")
    function: dict[str, Any] = {
        "name": name,
        "description": tool.get("description", ""),
        "parameters": parameters,
    }
    if "strict" in tool:
        function["strict"] = tool["strict"]
    return {"type": tool.get("type", "function"), "function": function}
