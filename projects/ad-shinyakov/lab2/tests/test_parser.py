import json
import tempfile
import unittest
from pathlib import Path

import mistune

from parser.document import DocumentParser
from parser.images import ImageStore
from parser.source import RepositorySource
from parser.storage import FileStorage


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "website"
        self.docs = self.repo / "content/en/docs"
        self.docs.mkdir(parents=True)
        self.source = RepositorySource(self.repo)
        self.storage = FileStorage(self.root / "output")
        self.parser = DocumentParser(self.source, self.storage, ImageStore(self.source, self.storage))

    def tearDown(self):
        self.temporary.cleanup()

    def parse(self, body):
        path = self.docs / "example.md"
        path.write_text('---\ntitle: Example\ndraft: false\n---\n' + body)
        metadata = self.parser.parse(path)
        _, text = self.source.read(self.storage.root / metadata["path"])
        return text, metadata

    def test_reference_links_and_parentheses(self):
        text, metadata = self.parse('[Reference][ref]\n\n[ref]: /docs/other/\n\n[Nested](https://example.org/a_(b))\n')
        self.assertEqual([link["url"] for link in metadata["links"]], [
            'https://kubernetes.io/docs/other/', 'https://example.org/a_(b)',
        ])
        self.assertIn('https://kubernetes.io/docs/other/', text)
        self.assertIn('https://example.org/a_(b)', text)

    def test_inline_and_fenced_code_are_not_rewritten(self):
        code = '[literal](/docs/keep/) {{< note >}}'
        text, metadata = self.parse(f'`{code}`\n\n```yaml\nspec:\n  text: "{code}"\n```\n')
        self.assertEqual(metadata["links"], [])
        self.assertEqual(metadata["warnings"], [])
        self.assertIn(f'`{code}`', text)
        self.assertIn(f'  text: "{code}"', text)

    def test_code_inside_lists_and_quotes_preserves_structure(self):
        body = '> ```yaml\n> spec:\n>   value: "{{< note >}}"\n> ```\n\n- Example:\n\n  ```yaml\n  spec:\n    value: "{{% code_sample file=missing %}}"\n  ```\n'
        text, metadata = self.parse(body)
        self.assertEqual(metadata["warnings"], [])
        tokens = mistune.create_markdown(renderer=None)(text)
        quote = next(token for token in tokens if token["type"] == "block_quote")
        self.assertIn('  value: "{{< note >}}"', quote["children"][0]["raw"])
        self.assertTrue(any(token["type"] == "list" for token in tokens))

    def test_shortcodes_expand_and_generated_code_stays_literal(self):
        examples = self.repo / "content/en/examples"
        examples.mkdir()
        (examples / "pod.yaml").write_text('kind: Pod\n  example: "[literal](/keep/) {{< note >}}"\n')
        text, metadata = self.parse('{{< note >}}\nExplanation\n{{< /note >}}\n\n{{% code_sample file="pod.yaml" %}}\n\n[Other]({{< ref "/docs/other.md" >}})\n')
        self.assertIn('**Note:**', text)
        self.assertIn('kind: Pod', text)
        self.assertIn('  example: "[literal](/keep/) {{< note >}}"', text)
        self.assertEqual(metadata["warnings"], [])
        self.assertEqual(metadata["links"][0]["url"], 'https://kubernetes.io/docs/other')

    def test_tables_and_lists_survive(self):
        body = '| Name | Value |\n| :--- | ---: |\n| port | `80` |\n\n- First\n- Second\n\n- [x] Done\n\n~~old~~\n'
        text, _ = self.parse(body)
        tokens = mistune.create_markdown(renderer=None, plugins=['table', 'task_lists', 'strikethrough'])(text)
        self.assertTrue(any(token["type"] == 'table' for token in tokens))
        self.assertIn('`80`', text)
        self.assertIn('~~old~~', text)
        self.assertIn('[x]', text)

    def test_images_from_markdown_and_embedded_html(self):
        assets = self.repo / "static/images"
        assets.mkdir(parents=True)
        (assets / "scheme.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        text, metadata = self.parse('![Scheme](/images/scheme.svg "Caption")\n\n<figure><img src="/images/scheme.svg" alt="HTML scheme"><figcaption>HTML caption</figcaption></figure>\n')
        self.assertEqual(len(metadata["images"]), 2)
        self.assertEqual(len(list((self.storage.root / "assets").iterdir())), 1)
        self.assertEqual(metadata["images"][0]["caption"], 'Caption')
        self.assertIn('HTML caption', text)
        self.assertIn('../assets/', text)

    def test_unknown_shortcode_preserved_and_comments_ignored(self):
        text, metadata = self.parse('<!-- {{< hidden >}} -->\n\n{{< unknown value="x" >}}\n')
        self.assertIn('{{< unknown value="x" >}}', text)
        self.assertNotIn('hidden', text)
        self.assertEqual(metadata['warnings'], ['Unsupported shortcode preserved: unknown'])

    def test_front_matter_round_trip(self):
        _, metadata = self.parse('Body\n')
        path = self.storage.root / metadata["path"]
        saved, body = RepositorySource.read(path)
        self.assertEqual(saved['title'], 'Example')
        self.assertIs(saved['source_dirty'], False)
        self.assertEqual(metadata['front_matter']['draft'], False)
        self.assertIn('Body', body)
        self.assertEqual(json.loads(path.with_suffix('.json').read_text())['source_url'], saved['source_url'])
