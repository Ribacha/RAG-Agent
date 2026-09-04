from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rag_agent.ingest.pipeline import discover_files, ingest_path, load_file
from rag_agent.ingest.text import read_text
from rag_agent.models import IngestionError


class IngestionTests(unittest.TestCase):
    def test_gb18030_text_is_decoded_without_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.txt"
            path.write_bytes("中文内容".encode("gb18030"))

            result = read_text(path)

            self.assertEqual(result.text, "中文内容")
            self.assertNotIn("\ufffd", result.text)
            self.assertIn(result.encoding.lower().replace("-", ""), {"gb18030", "gb2312"})

    def test_utf16_bom_text_is_decoded_before_binary_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "utf16.txt"
            path.write_bytes("第一行\n第二行".encode("utf-16"))

            result = read_text(path)

            self.assertEqual(result.text, "第一行\n第二行")
            self.assertEqual(result.encoding, "utf-16")

    def test_bomless_utf16_ascii_detects_both_byte_orders(self) -> None:
        for encoding in ("utf-16-le", "utf-16-be"):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "ascii.txt"
                path.write_bytes("hello world\nsecond line".encode(encoding))

                result = read_text(path)

                self.assertEqual(result.text, "hello world\nsecond line")
                self.assertEqual(result.encoding, "utf-16-heuristic")
                self.assertIn("encoding_detected_by_utf16_heuristic", result.warnings)

    def test_bomless_utf16_single_ascii_code_unit_is_detected(self) -> None:
        for encoding in ("utf-16-le", "utf-16-be"):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "one-character.txt"
                path.write_bytes("a".encode(encoding))

                result = read_text(path)

                self.assertEqual(result.text, "a")
                self.assertEqual(result.encoding, "utf-16-heuristic")

    def test_bomless_utf16_cjk_with_newline_is_not_treated_as_binary(self) -> None:
        for encoding in ("utf-16-le", "utf-16-be"):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cjk.txt"
                path.write_bytes("第一行\n第二行".encode(encoding))

                result = read_text(path)

                self.assertEqual(result.text, "第一行\n第二行")
                self.assertEqual(result.encoding, "utf-16-heuristic")

    def test_bomless_utf16_mixed_cjk_and_ascii_is_decoded(self) -> None:
        for encoding in ("utf-16-le", "utf-16-be"):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "mixed.txt"
                path.write_bytes("中文abc\n测试".encode(encoding))

                result = read_text(path)

                self.assertEqual(result.text, "中文abc\n测试")

    def test_same_content_different_paths_have_distinct_source_and_document_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "one.txt"
            second_path = root / "two.txt"
            first_path.write_text("相同内容", encoding="utf-8")
            second_path.write_text("相同内容", encoding="utf-8")

            first = load_file(first_path)
            second = load_file(second_path)

            self.assertEqual(first.content_hash, second.content_hash)
            self.assertNotEqual(first.source_id, second.source_id)
            self.assertNotEqual(first.doc_id, second.doc_id)
            self.assertEqual(first.blocks[0].source_id, first.source_id)

    def test_normalized_offsets_are_explicit_for_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "newlines.txt"
            path.write_bytes("甲\r\n乙".encode("utf-8"))

            record = load_file(path)
            block = record.blocks[0]

            self.assertEqual(block.text, "甲\n乙")
            self.assertEqual((block.normalized_char_start, block.normalized_char_end), (0, 3))
            # Legacy names remain aliases for callers written against 0.1.
            self.assertEqual(block.source_char_start, block.normalized_char_start)
            self.assertEqual(block.source_char_end, block.normalized_char_end)
    def test_utf8_bom_txt_is_read_without_bom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_bytes("\ufeff第一行\n\n第二行".encode("utf-8"))

            record = load_file(path)

            self.assertEqual(record.file_type, "txt")
            self.assertEqual(record.blocks[0].text, "第一行\n\n第二行")
            self.assertEqual(record.blocks[0].extraction_method, "txt")

    def test_markdown_keeps_heading_path_and_code_fence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.md"
            path.write_text(
                "# Transformer\n\n介绍。\n\n"
                "## Attention\n\n```python\nprint('x')\n\n```\n",
                encoding="utf-8",
            )

            record = load_file(path)

            self.assertEqual(len(record.blocks), 4)
            self.assertEqual(record.blocks[0].heading_path, ("Transformer",))
            self.assertEqual(record.blocks[1].heading_path, ("Transformer",))
            self.assertEqual(
                record.blocks[2].heading_path,
                ("Transformer", "Attention"),
            )
            self.assertIn("print('x')", record.blocks[3].text)

    def test_markdown_normalized_offsets_point_to_block_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offsets.md"
            path.write_bytes("# 标题\r\n\r\n正文第一行\r\n正文第二行\r\n".encode("utf-8"))

            record = load_file(path)
            normalized = record.blocks[0].text
            start = record.blocks[0].normalized_char_start
            end = record.blocks[0].normalized_char_end

            self.assertEqual(normalized, "# 标题")
            self.assertEqual(record.blocks[0].source_char_start, start)
            self.assertEqual(record.blocks[0].source_char_end, end)
            self.assertEqual(record.blocks[0].text, record.blocks[0].text.strip())

            body = record.blocks[1]
            normalized_source = record.blocks[0].text + "\n\n" + body.text
            self.assertEqual(normalized_source[start:end], normalized)

    def test_discover_files_filters_extensions_and_hidden_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.md").write_text("a", encoding="utf-8")
            (root / "b.txt").write_text("b", encoding="utf-8")
            (root / "c.docx").write_bytes(b"ignored")
            hidden = root / ".hidden"
            hidden.mkdir()
            (hidden / "secret.txt").write_text("ignored", encoding="utf-8")

            found = discover_files(root)

            self.assertEqual([path.name for path in found], ["a.md", "b.txt"])

    def test_empty_directory_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(IngestionError):
                discover_files(Path(directory))

    def test_batch_keeps_other_files_when_pdf_dependency_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ok.txt").write_text("可用文本", encoding="utf-8")
            (root / "broken.pdf").write_bytes(b"not a real pdf")

            batch = ingest_path(root)

            self.assertEqual(len(batch.records), 1)
            self.assertEqual(batch.records[0].file_type, "txt")
            self.assertEqual(len(batch.failures), 1)
            self.assertEqual(
                batch.failures[0].source_path,
                str((root / "broken.pdf").resolve()),
            )
            self.assertIn("PDF", batch.failures[0].message)


if __name__ == "__main__":
    unittest.main()
