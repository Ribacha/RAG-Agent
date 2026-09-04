from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rag_agent.chunking.splitter import ChunkConfig
from rag_agent.ingest.incremental import incremental_ingest
from rag_agent.ingest.pipeline import ingest_path
from rag_agent.storage.jsonl import write_jsonl_atomic


class IncrementalIngestionTests(unittest.TestCase):
    def test_unchanged_document_reuses_previous_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "note.txt"
            path.write_text("保持不变", encoding="utf-8")
            initial = ingest_path(path)
            from rag_agent.chunking.splitter import chunk_document
            record = initial.records[0]
            chunks = [chunk.to_dict() for chunk in chunk_document(record)]
            manifests = [record.to_manifest_dict(len(chunks), chunking_fingerprint=ChunkConfig().fingerprint())]

            result = incremental_ingest(
                path,
                existing_chunks=chunks,
                existing_manifests=manifests,
            )

            self.assertEqual(result.documents_unchanged, 1)
            self.assertEqual(result.documents_updated, 0)
            self.assertEqual(result.chunks_generated, 0)
            self.assertEqual(result.chunks_reused, len(chunks))
            self.assertEqual(result.chunks, tuple(chunks))

    def test_reuses_unchanged_updates_changed_and_removes_deleted_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "docs"
            root.mkdir()
            first_path = root / "first.txt"
            second_path = root / "second.txt"
            first_path.write_text("原始内容", encoding="utf-8")
            second_path.write_text("将被删除", encoding="utf-8")
            initial = ingest_path(root)
            old_chunks = []
            old_manifests = []
            from rag_agent.chunking.splitter import chunk_document
            for record in initial.records:
                rows = [chunk.to_dict() for chunk in chunk_document(record)]
                old_chunks.extend(rows)
                old_manifests.append(record.to_manifest_dict(len(rows), chunking_fingerprint=ChunkConfig().fingerprint()))

            first_path.write_text("修改后的内容", encoding="utf-8")
            second_path.unlink()
            result = incremental_ingest(
                root,
                existing_chunks=old_chunks,
                existing_manifests=old_manifests,
            )

            self.assertEqual(result.documents_added, 0)
            self.assertEqual(result.documents_updated, 1)
            self.assertEqual(result.documents_unchanged, 0)
            self.assertEqual(result.documents_deleted, 1)
            self.assertEqual(result.documents_failed, 0)
            self.assertEqual({row["source_path"] for row in result.manifests}, {str(first_path.resolve())})
            self.assertIn("修改后的内容", result.chunks[0]["text"])

    def test_parse_failure_keeps_last_known_good_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "docs"
            root.mkdir()
            path = root / "note.txt"
            path.write_text("可用版本", encoding="utf-8")
            initial = ingest_path(path)
            from rag_agent.chunking.splitter import chunk_document
            record = initial.records[0]
            chunks = [chunk.to_dict() for chunk in chunk_document(record)]
            manifests = [record.to_manifest_dict(len(chunks), chunking_fingerprint=ChunkConfig().fingerprint())]

            path.write_bytes(b"\x00\x00\x00\x00")
            result = incremental_ingest(path, existing_chunks=chunks, existing_manifests=manifests)

            self.assertEqual(result.documents_failed, 1)
            self.assertEqual(len(result.chunks), 1)
            self.assertEqual(result.chunks[0]["text"], "可用版本")
