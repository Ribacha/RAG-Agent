"""Bounded, JSONL-persisted conversation history for Agent runs.

Only completed user/assistant turns are persisted.  Tool-call messages belong to
one run and are intentionally excluded from the next run's prompt: the Agent
can search again when a follow-up needs fresh evidence, while the history file
stays small and cannot contain a dangling tool-call protocol sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..storage.jsonl import read_jsonl, write_jsonl_atomic


HISTORY_SCHEMA_VERSION = 1
DEFAULT_MAX_TURNS = 20
MAX_ALLOWED_TURNS = 100


@dataclass(frozen=True)
class ConversationTurn:
    """One completed user question and assistant answer."""

    user: str
    assistant: str

    def __post_init__(self) -> None:
        user = _clean_text(self.user, "user")
        assistant = _clean_text(self.assistant, "assistant")
        object.__setattr__(self, "user", user)
        object.__setattr__(self, "assistant", assistant)

    def to_dict(self) -> dict[str, str]:
        return {"user": self.user, "assistant": self.assistant}

    @classmethod
    def from_dict(cls, value: Any, *, position: int) -> "ConversationTurn":
        if not isinstance(value, dict):
            raise ValueError(f"历史第 {position} 轮必须是对象")
        try:
            return cls(user=value["user"], assistant=value["assistant"])
        except KeyError as error:
            raise ValueError(f"历史第 {position} 轮缺少字段：{error.args[0]}") from error
        except ValueError as error:
            raise ValueError(f"历史第 {position} 轮：{error}") from error


@dataclass(frozen=True)
class ConversationHistory:
    """A bounded immutable history that can be passed between Agent turns."""

    turns: tuple[ConversationTurn, ...] = ()
    max_turns: int = DEFAULT_MAX_TURNS

    def __post_init__(self) -> None:
        _validate_max_turns(self.max_turns)
        if not isinstance(self.turns, (tuple, list)):
            raise ValueError("history turns 必须是序列")
        normalised = tuple(
            turn if isinstance(turn, ConversationTurn) else ConversationTurn.from_dict(turn, position=position)
            for position, turn in enumerate(self.turns, start=1)
        )
        if len(normalised) > self.max_turns:
            normalised = normalised[-self.max_turns :]
        object.__setattr__(self, "turns", normalised)

    @property
    def size(self) -> int:
        return len(self.turns)

    def append(self, user: str, assistant: str) -> "ConversationHistory":
        """Return a new history containing the latest completed turn."""

        turn = ConversationTurn(user=user, assistant=assistant)
        return ConversationHistory(
            turns=(*self.turns, turn),
            max_turns=self.max_turns,
        )

    def to_messages(self) -> list[dict[str, str]]:
        """Convert turns to chat messages without exposing persistence fields."""

        messages: list[dict[str, str]] = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        return messages

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "max_turns": self.max_turns,
            "turns": [turn.to_dict() for turn in self.turns],
        }

    def save(self, path: Path) -> None:
        """Atomically replace a JSONL history file with this snapshot."""

        write_jsonl_atomic(path, [self.to_dict()])

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        default_max_turns: int = DEFAULT_MAX_TURNS,
    ) -> "ConversationHistory":
        """Restore a validated history snapshot from an in-memory mapping."""

        _validate_max_turns(default_max_turns)
        if not isinstance(value, Mapping):
            raise ValueError("历史快照必须是对象")
        if value.get("schema_version") != HISTORY_SCHEMA_VERSION:
            raise ValueError(
                f"不支持的历史版本：{value.get('schema_version')}，期望 {HISTORY_SCHEMA_VERSION}"
            )
        stored_max_turns = value.get("max_turns", default_max_turns)
        if isinstance(stored_max_turns, bool) or not isinstance(stored_max_turns, int):
            raise ValueError("历史 max_turns 必须是整数")
        _validate_max_turns(stored_max_turns)
        raw_turns = value.get("turns", [])
        if not isinstance(raw_turns, list):
            raise ValueError("历史 turns 必须是数组")
        turns = tuple(
            ConversationTurn.from_dict(item, position=position)
            for position, item in enumerate(raw_turns, start=1)
        )
        return cls(turns=turns, max_turns=stored_max_turns)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        missing_ok: bool = True,
    ) -> "ConversationHistory":
        """Load one history snapshot; a missing file starts an empty history."""

        _validate_max_turns(max_turns)
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            if missing_ok:
                return cls(max_turns=max_turns)
            raise FileNotFoundError(f"历史文件不存在：{resolved}")
        rows = list(read_jsonl(resolved))
        if len(rows) != 1:
            raise ValueError("历史文件必须恰好包含一个 JSONL 快照对象")
        # The persisted limit wins, so loading cannot unexpectedly expand a
        # history that was deliberately bounded when it was written.
        return cls.from_dict(rows[0], default_max_turns=max_turns)


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是字符串")
    clean = value.strip()
    if not clean:
        raise ValueError(f"{field_name} 不能为空")
    return clean


def _validate_max_turns(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_turns 必须是整数")
    if not 1 <= value <= MAX_ALLOWED_TURNS:
        raise ValueError(f"max_turns 必须在 1 到 {MAX_ALLOWED_TURNS} 之间")
