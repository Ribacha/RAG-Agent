from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rag_agent.agent import ConversationHistory, ConversationTurn


class ConversationHistoryTests(unittest.TestCase):
    def test_append_trims_oldest_turn_and_builds_chat_messages(self) -> None:
        history = ConversationHistory(max_turns=2)
        history = history.append("第一个问题", "第一个答案")
        history = history.append("第二个问题", "第二个答案")
        history = history.append("第三个问题", "第三个答案")

        self.assertEqual(history.size, 2)
        self.assertEqual(history.turns[0].user, "第二个问题")
        self.assertEqual(
            history.to_messages(),
            [
                {"role": "user", "content": "第二个问题"},
                {"role": "assistant", "content": "第二个答案"},
                {"role": "user", "content": "第三个问题"},
                {"role": "assistant", "content": "第三个答案"},
            ],
        )

    def test_save_and_load_round_trip_is_jsonl_and_bounded(self) -> None:
        history = ConversationHistory(
            turns=(
                ConversationTurn("问题", "答案"),
            ),
            max_turns=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            history.save(path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["schema_version"], 1)
            loaded = ConversationHistory.load(path)
            self.assertEqual(loaded, history)

    def test_invalid_or_empty_history_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_turns"):
            ConversationHistory(max_turns=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "恰好包含一个"):
                ConversationHistory.load(path)

            path.write_text(
                json.dumps({"schema_version": 1, "max_turns": 2, "turns": [{"user": "x"}]})
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "缺少字段"):
                ConversationHistory.load(path)


if __name__ == "__main__":
    unittest.main()
