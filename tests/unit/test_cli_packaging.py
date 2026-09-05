from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_agent import __version__
from rag_agent import cli as cli_module
from rag_agent.answering.chat import ToolChatTurn
from rag_agent.cli import main
from rag_agent.embeddings import HashEmbeddingProvider
from rag_agent.retrieval import build_vector_index
from rag_agent.workspace import (
    WORKSPACE_DIR,
    create_workspace,
    find_workspace_root,
    paths_for,
)


class InitCommandTests(unittest.TestCase):
    def test_init_creates_workspace_structure_and_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "kb"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(["init", str(root), "--api-key", "sk-test", "--no-input"])

            self.assertEqual(exit_code, 0)
            paths = paths_for(root)
            for path in (
                paths.inbox,
                paths.chunks.parent,
                paths.failures.parent,
                paths.state,
                root / WORKSPACE_DIR,
            ):
                self.assertTrue(path.is_dir(), path)
            env_text = paths.env_file.read_text(encoding="utf-8")
            self.assertIn("LLM_API_KEY=sk-test", env_text)
            self.assertIn("LLM_BASE_URL=https://api.deepseek.com", env_text)
            self.assertIn("EMBEDDING_PROVIDER=hash", env_text)
            meta = json.loads(
                (root / WORKSPACE_DIR / "workspace.json").read_text(encoding="utf-8")
            )
            self.assertEqual(meta["schema_version"], 1)

    def test_init_keeps_existing_env_unless_forced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "kb"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["init", str(root), "--api-key", "sk-first", "--no-input"])
            with contextlib.redirect_stdout(io.StringIO()):
                main(["init", str(root), "--api-key", "sk-second", "--no-input"])
            env_text = paths_for(root).env_file.read_text(encoding="utf-8")
            self.assertIn("LLM_API_KEY=sk-first", env_text)

            with contextlib.redirect_stdout(io.StringIO()):
                main(["init", str(root), "--api-key", "sk-third", "--no-input", "--force"])
            env_text = paths_for(root).env_file.read_text(encoding="utf-8")
            self.assertIn("LLM_API_KEY=sk-third", env_text)

    def test_init_defaults_to_current_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = main(["init", "--no-input"])
            finally:
                os.chdir(previous)
            self.assertEqual(exit_code, 0)
            self.assertTrue((root / WORKSPACE_DIR).is_dir())


class WorkspaceDiscoveryTests(unittest.TestCase):
    def test_find_workspace_root_from_nested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "kb"
            create_workspace(root)
            nested = root / "data" / "inbox"
            nested.mkdir(parents=True, exist_ok=True)
            self.assertEqual(find_workspace_root(start=nested), root.resolve())

    def test_find_workspace_root_honors_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            override = Path(directory) / "override"
            override.mkdir()
            elsewhere = Path(directory) / "elsewhere"
            create_workspace(elsewhere)
            with patch.dict(os.environ, {"RAG_AGENT_ROOT": str(override)}):
                self.assertEqual(find_workspace_root(start=elsewhere), override.resolve())


class DoctorCommandTests(unittest.TestCase):
    def test_doctor_json_report_for_fresh_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "kb"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["init", str(root), "--no-input"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch.dict(
                os.environ, {"RAG_AGENT_ROOT": str(root.resolve())}
            ):
                exit_code = main(["doctor", "--json"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["workspace_root"], str(root.resolve()))
            names = {check["name"] for check in payload["checks"]}
            self.assertIn("工作区", names)
            self.assertIn("数据目录", names)
            self.assertIn("向量索引", names)
            statuses = {check["status"] for check in payload["checks"]}
            # 全新工作区只有警告（缺 Key、缺索引等），不应出现错误项。
            self.assertNotIn("error", statuses)

    def test_doctor_flags_broken_index_as_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "kb"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["init", str(root), "--no-input"])
            paths = paths_for(root)
            paths.index.parent.mkdir(parents=True, exist_ok=True)
            paths.index.write_text("not-meta\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch.dict(
                os.environ, {"RAG_AGENT_ROOT": str(root.resolve())}
            ):
                exit_code = main(["doctor", "--json"])
            self.assertEqual(exit_code, 1)
            payload = json.loads(output.getvalue())
            index_checks = [
                check for check in payload["checks"] if check["name"] == "向量索引"
            ]
            self.assertEqual(index_checks[0]["status"], "error")


class VersionCommandTests(unittest.TestCase):
    def test_version_subcommand_prints_version(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["version"])
        self.assertEqual(exit_code, 0)
        self.assertIn(__version__, output.getvalue())

    def test_version_flag_exits_cleanly(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn(__version__, output.getvalue())


class ChatCommandTests(unittest.TestCase):
    def _build_index(self, index_path: Path) -> None:
        build_vector_index(
            [
                {
                    "chunk_id": "tcp-handshake",
                    "doc_id": "doc",
                    "source_path": "/docs/network.txt",
                    "file_type": "txt",
                    "text": "TCP 通过三次握手建立连接。",
                }
            ],
            provider=HashEmbeddingProvider(dimension=64),
            path=index_path,
        )

    def test_chat_retrieval_only_answers_from_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "vectors.jsonl"
            self._build_index(index_path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch(
                "sys.stdin", new=io.StringIO("三次握手\n/search 三次握手\nexit\n")
            ):
                exit_code = main(
                    [
                        "chat",
                        "--index",
                        str(index_path),
                        "--retrieval-only",
                        "--embedding-dimension",
                        "64",
                        "--min-score",
                        "-1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            rendered = output.getvalue()
            self.assertIn("离线检索", rendered)
            self.assertGreaterEqual(rendered.count("score="), 2)

    def test_chat_agent_mode_persists_history(self) -> None:
        class HistoryChat:
            model = "fake-chat-model"

            def complete_with_tools(self, messages, tools):
                return ToolChatTurn(
                    content="历史已被记录的答案。",
                    tool_calls=(),
                    assistant_message={"role": "assistant", "content": "历史已被记录的答案。"},
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "vectors.jsonl"
            history_path = root / "history.jsonl"
            self._build_index(index_path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch(
                "sys.stdin", new=io.StringIO("第一问\nexit\n")
            ), patch(
                "rag_agent.cli.OpenAICompatibleChatProvider.from_environment",
                return_value=HistoryChat(),
            ):
                exit_code = main(
                    [
                        "chat",
                        "--index",
                        str(index_path),
                        "--agent",
                        "--history",
                        str(history_path),
                        "--embedding-dimension",
                        "64",
                        "--llm-api-key",
                        "test-key",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("历史已被记录的答案。", output.getvalue())
            history = json.loads(history_path.read_text(encoding="utf-8"))
            self.assertEqual(history["turns"][-1]["user"], "第一问")


if __name__ == "__main__":
    unittest.main()
