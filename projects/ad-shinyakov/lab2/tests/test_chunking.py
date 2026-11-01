import tempfile
import unittest
from pathlib import Path

from chunker import ChunkConfig, ChunkingPipeline, MarkdownChunker
from chunker.cli import parse_arguments


class CharacterCounter:
    model = "test-characters"

    def count(self, text):
        return len(text)


class ChunkingTests(unittest.TestCase):
    sample = "# Doc\n\nExplanation.\n\n```yaml\nspec:\n  replicas: 3\n```\n\nAfter code.\n\n## Next\n\nOther section.\n"

    def test_related_fragments_share_section_and_numbering(self):
        chunks = MarkdownChunker(CharacterCounter()).split(self.sample, "doc", {})
        self.assertEqual(len(chunks), 4)
        self.assertEqual(len({chunk["section_id"] for chunk in chunks[:3]}), 1)
        self.assertNotEqual(chunks[2]["section_id"], chunks[3]["section_id"])
        self.assertEqual([chunk["section_part"] for chunk in chunks[:3]], [0, 1, 2])
        self.assertTrue(all(chunk["section_parts"] == 3 for chunk in chunks[:3]))
        self.assertEqual(chunks[1]["code_language"], "yaml")
        self.assertIn("  replicas: 3", chunks[1]["text"])
        self.assertEqual(len({chunk["id"] for chunk in chunks}), len(chunks))
        self.assertTrue(all("config" not in chunk and "tokenizer" not in chunk for chunk in chunks))
        self.assertEqual(chunks[0]["next_chunk_id"], chunks[1]["id"])
        self.assertEqual(chunks[1]["previous_chunk_id"], chunks[0]["id"])

    def test_indentation_and_heading_budget(self):
        chunker = MarkdownChunker(CharacterCounter(), ChunkConfig(100, 0, True))
        body = "  field: value\n" * 30
        parts = chunker._split_section(body, "Doc\n\n")
        self.assertEqual("".join(parts), body)
        self.assertTrue(all(part.startswith(("  ", "\n  ")) for part in parts))
        self.assertTrue(all(len("Doc\n\n" + part) <= 100 for part in parts))

    def test_recursive_mode_and_cli(self):
        args = parse_arguments(["--splitter", "recursive"])
        self.assertEqual(args.splitter, "recursive")
        self.assertFalse(args.include_headings)
        chunks = MarkdownChunker(CharacterCounter(), ChunkConfig(splitter="recursive")).split(self.sample, "doc", {"title": "Document"})
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], self.sample)
        self.assertEqual(chunks[0]["heading_path"], ["Document"])
        with self.assertRaises(ValueError):
            ChunkConfig(splitter="unknown")

    def test_whitespace_only_fragments_are_skipped(self):
        for headings in (False, True):
            chunker = MarkdownChunker(CharacterCounter(), ChunkConfig(100, 0, headings))
            chunks = chunker.split("# Doc\n\n" + "x" * 200 + "\n\n", "doc", {})
            self.assertTrue(chunks)
            self.assertTrue(all(chunk["text"].strip() for chunk in chunks))
            self.assertTrue(all("x" in chunk["text"] for chunk in chunks))
            self.assertEqual([chunk["section_part"] for chunk in chunks], list(range(len(chunks))))
            self.assertTrue(all(chunk["section_parts"] == len(chunks) for chunk in chunks))
            self.assertIsNone(chunks[-1]["next_chunk_id"])

    def test_modes_have_distinct_reproducible_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            source.mkdir()
            (source / "doc.md").write_text(self.sample)
            paths = []
            for strategy in ("markdown", "recursive"):
                pipeline = ChunkingPipeline(MarkdownChunker(CharacterCounter(), ChunkConfig(splitter=strategy)), root / "output")
                first, _ = pipeline.run(source)
                second, _ = pipeline.run(source)
                self.assertEqual(first, second)
                paths.append(first)
            self.assertNotEqual(*paths)
