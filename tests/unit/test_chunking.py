from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from rag_agent.chunking.splitter import (
    ChunkConfig,
    chunk_document,
    split_text_with_offsets,
)
from rag_agent.ingest.pipeline import load_file
from rag_agent.models import DocumentRecord, TextBlock


class ChunkingTests(unittest.TestCase):
    def test_long_text_has_overlap_and_offsets(self) -> None:
        text = "甲乙丙丁戊己庚辛壬癸" * 4
        pieces = split_text_with_offsets(
            text,
            max_chars=12,
            overlap_chars=3,
        )

        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(piece) <= 12 for piece, _, _ in pieces))
        self.assertEqual(pieces[0][0][-3:], pieces[1][0][:3])
        self.assertEqual(pieces[0][1], 0)
        self.assertEqual(pieces[-1][2], len(text))

    def test_chunk_id_is_deterministic(self) -> None:
        block = TextBlock(
            doc_id="doc",
            source_path="/tmp/a.txt",
            file_type="txt",
            text="第一段\n第二段",
            extraction_method="txt",
            source_char_start=0,
            source_char_end=7,
        )
        document = DocumentRecord(
            doc_id="doc",
            source_path="/tmp/a.txt",
            file_type="txt",
            content_hash="file-hash",
            size_bytes=20,
            blocks=[block],
        )

        first = chunk_document(document, config=ChunkConfig(max_chars=5, overlap_chars=1))
        second = chunk_document(document, config=ChunkConfig(max_chars=5, overlap_chars=1))

        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])
        self.assertTrue(all(chunk.doc_id == "doc" for chunk in first))
        self.assertTrue(all(chunk.source_path == "/tmp/a.txt" for chunk in first))
        self.assertEqual(first[0].chunking_fingerprint, second[0].chunking_fingerprint)
        self.assertEqual(first[0].normalized_char_start, first[0].char_start)
        self.assertEqual(first[0].normalized_char_end, first[0].char_end)

    def test_chunk_configuration_fingerprint_changes_chunk_ids(self) -> None:
        block = TextBlock(
            doc_id="doc",
            source_path="/tmp/config.txt",
            file_type="txt",
            text="甲乙丙丁戊己庚辛壬癸",
            extraction_method="txt",
            normalized_char_start=0,
            normalized_char_end=10,
        )
        document = DocumentRecord(
            doc_id="doc",
            source_path="/tmp/config.txt",
            file_type="txt",
            content_hash="file-hash",
            size_bytes=20,
            blocks=[block],
            ingestion_fingerprint="parser-fingerprint",
        )

        small = chunk_document(document, config=ChunkConfig(max_chars=5, overlap_chars=1))
        large = chunk_document(document, config=ChunkConfig(max_chars=7, overlap_chars=1))

        self.assertNotEqual(
            small[0].chunking_fingerprint,
            large[0].chunking_fingerprint,
        )
        self.assertNotEqual(
            [chunk.chunk_id for chunk in small],
            [chunk.chunk_id for chunk in large],
        )
        self.assertEqual(document.chunking_fingerprint, large[0].chunking_fingerprint)

    def test_same_content_different_sources_have_distinct_chunk_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "one.txt"
            second_path = root / "two.txt"
            first_path.write_text("可检索的相同内容", encoding="utf-8")
            second_path.write_text("可检索的相同内容", encoding="utf-8")

            first_chunks = chunk_document(load_file(first_path))
            second_chunks = chunk_document(load_file(second_path))

            self.assertEqual(first_chunks[0].content_hash, second_chunks[0].content_hash)
            self.assertNotEqual(first_chunks[0].source_id, second_chunks[0].source_id)
            self.assertNotEqual(first_chunks[0].chunk_id, second_chunks[0].chunk_id)

    def test_page_and_heading_metadata_are_carried_to_chunk(self) -> None:
        block = TextBlock(
            doc_id="doc",
            source_path="/tmp/a.pdf",
            file_type="pdf",
            text="页面内容",
            extraction_method="pdf_text",
            page_number=3,
            heading_path=("章节",),
        )
        document = DocumentRecord(
            doc_id="doc",
            source_path="/tmp/a.pdf",
            file_type="pdf",
            content_hash="hash",
            size_bytes=10,
            blocks=[block],
        )

        chunk = chunk_document(document)[0]

        self.assertEqual(chunk.page_start, 3)
        self.assertEqual(chunk.page_end, 3)
        self.assertEqual(chunk.heading_path, ("章节",))
        self.assertEqual(chunk.extraction_methods, ("pdf_text",))


if __name__ == "__main__":
    unittest.main()
