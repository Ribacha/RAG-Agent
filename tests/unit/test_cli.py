from __future__ import annotations

import json
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag_agent.answering.chat import ToolChatTurn
from rag_agent import cli as cli_module
from rag_agent.cli import main
from rag_agent.embeddings import HashEmbeddingProvider
from rag_agent.retrieval import build_vector_index


class CliTests(unittest.TestCase):
    def test_local_env_is_loaded_without_overriding_shell_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "RAG_AGENT_TEST_DOTENV=from-file\n"
                "RAG_AGENT_TEST_DOTENV_ONLY=loaded\n",
                encoding="utf-8",
            )
            with patch.object(cli_module, "PROJECT_ROOT", root), patch.dict(
                os.environ,
                {"RAG_AGENT_TEST_DOTENV": "from-shell"},
                clear=False,
            ):
                os.environ.pop("RAG_AGENT_TEST_DOTENV_ONLY", None)
                cli_module._load_local_env()
                self.assertEqual(os.environ["RAG_AGENT_TEST_DOTENV"], "from-shell")
                self.assertEqual(os.environ["RAG_AGENT_TEST_DOTENV_ONLY"], "loaded")

    def test_ingest_writes_chunks_manifest_and_failure_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "a.md").write_text("# 标题\n\n正文", encoding="utf-8")
            output = root / "out" / "chunks.jsonl"
            manifest = root / "out" / "documents.jsonl"
            failures = root / "out" / "failures.jsonl"

            exit_code = main(
                [
                    "ingest",
                    str(source),
                    "--output",
                    str(output),
                    "--manifest",
                    str(manifest),
                    "--failures",
                    str(failures),
                ]
            )

            self.assertEqual(exit_code, 0)
            chunk_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            manifest_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertGreaterEqual(len(chunk_rows), 1)
            self.assertEqual(manifest_rows[0]["chunk_count"], len(chunk_rows))
            self.assertEqual(failures.read_text(encoding="utf-8"), "")

    def test_incremental_ingest_reports_reuse_and_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "a.txt").write_text("第一版", encoding="utf-8")
            (source / "b.txt").write_text("待删除", encoding="utf-8")
            output = root / "out" / "chunks.jsonl"
            manifest = root / "out" / "documents.jsonl"
            failures = root / "out" / "failures.jsonl"
            index = root / "out" / "vectors.jsonl"

            self.assertEqual(
                main(
                    [
                        "ingest",
                        str(source),
                        "--output",
                        str(output),
                        "--manifest",
                        str(manifest),
                        "--failures",
                        str(failures),
                        "--index",
                        str(index),
                    ]
                ),
                0,
            )
            (source / "a.txt").write_text("第二版", encoding="utf-8")
            (source / "b.txt").unlink()

            self.assertEqual(
                main(
                    [
                        "ingest",
                        str(source),
                        "--incremental",
                        "--output",
                        str(output),
                        "--manifest",
                        str(manifest),
                        "--failures",
                        str(failures),
                        "--index",
                        str(index),
                    ]
                ),
                0,
            )
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            chunks = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(rows[0]["source_path"], str((source / "a.txt").resolve()))

    def test_agent_history_flag_persists_and_reuses_completed_turns(self) -> None:
        class HistoryChat:
            model = "fake-agent-model"

            def __init__(self) -> None:
                self.messages = []

            def complete_with_tools(self, messages, tools):
                self.messages.append(list(messages))
                return ToolChatTurn(
                    content="可复用的答案。",
                    tool_calls=(),
                    assistant_message={"role": "assistant", "content": "可复用的答案。"},
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "vectors.jsonl"
            history_path = root / "history.jsonl"
            provider = HashEmbeddingProvider(dimension=64)
            build_vector_index(
                [
                    {
                        "chunk_id": "one",
                        "doc_id": "doc",
                        "source_path": "/docs/a.txt",
                        "file_type": "txt",
                        "text": "这是一个测试知识片段。",
                    }
                ],
                provider=provider,
                path=index_path,
            )
            chat = HistoryChat()
            with patch(
                "rag_agent.cli.OpenAICompatibleChatProvider.from_environment",
                return_value=chat,
            ):
                for question in ("第一问", "第二问"):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        exit_code = main(
                            [
                                "agent",
                                question,
                                "--index",
                                str(index_path),
                                "--history",
                                str(history_path),
                                "--embedding-dimension",
                                "64",
                                "--llm-api-key",
                                "test-key",
                                "--json",
                            ]
                        )
                    self.assertEqual(exit_code, 0)
                    self.assertEqual(json.loads(output.getvalue())["history"]["turns"][-1]["user"], question)

            self.assertEqual(len(chat.messages), 2)
            second_request = chat.messages[1]
            self.assertEqual(
                [(message["role"], message["content"]) for message in second_request],
                [
                    ("system", second_request[0]["content"]),
                    ("user", "第一问"),
                    ("assistant", "可复用的答案。"),
                    ("user", "第二问"),
                ],
            )
            self.assertEqual(json.loads(history_path.read_text(encoding="utf-8"))["turns"][-1]["user"], "第二问")

    def test_chinese_embedding_provider_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "network.txt").write_text(
                "TCP 通过三次握手建立连接。", encoding="utf-8"
            )
            output = root / "out" / "chunks.jsonl"
            manifest = root / "out" / "documents.jsonl"
            failures = root / "out" / "failures.jsonl"
            index = root / "out" / "vectors.jsonl"
            self.assertEqual(
                main(
                    [
                        "ingest",
                        str(source),
                        "--output",
                        str(output),
                        "--manifest",
                        str(manifest),
                        "--failures",
                        str(failures),
                        "--index",
                        str(index),
                        "--embedding-provider",
                        "chinese",
                        "--embedding-dimension",
                        "128",
                    ]
                ),
                0,
            )
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                self.assertEqual(
                    main(
                        [
                            "search",
                            "三次握手如何建立连接？",
                            "--index",
                            str(index),
                            "--embedding-provider",
                            "chinese",
                            "--embedding-dimension",
                            "128",
                            "--top-k",
                            "1",
                            "--min-score",
                            "-1",
                            "--json",
                        ]
                    ),
                    0,
                )
            payload = json.loads(captured.getvalue())
            self.assertEqual(payload[0]["source_path"], str((source / "network.txt").resolve()))

    def test_graph_flag_reports_optional_dependency_without_traceback(self) -> None:
        from rag_agent.agent.graph import langgraph_available

        if langgraph_available():
            self.skipTest("当前环境已安装 LangGraph")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "vectors.jsonl"
            provider = HashEmbeddingProvider(dimension=64)
            build_vector_index(
                [
                    {
                        "chunk_id": "one",
                        "doc_id": "doc",
                        "source_path": "/docs/a.txt",
                        "file_type": "txt",
                        "text": "测试片段。",
                    }
                ],
                provider=provider,
                path=index,
            )
            errors = io.StringIO()
            with contextlib.redirect_stderr(errors):
                exit_code = main(
                    [
                        "agent",
                        "测试问题",
                        "--graph",
                        "--index",
                        str(index),
                        "--embedding-dimension",
                        "64",
                        "--llm-api-key",
                        "test-key",
                    ]
                )
            self.assertEqual(exit_code, 2)
            self.assertIn("LangGraph 未安装", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
