import mistune
from bs4 import BeautifulSoup, Comment
from mistune.renderers.markdown import MarkdownRenderer


def html_soup(raw):
    soup = BeautifulSoup(raw, "html.parser")
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()
    return soup


class CodeMaskRenderer(MarkdownRenderer):
    def __init__(self, source):
        super().__init__()
        self.prefix = "PARSERPROTECTEDCODE"
        while self.prefix in source:
            self.prefix += "X"
        self.markers = {"{{<": self.prefix + "ANGLEEND", "{{%": self.prefix + "PERCENTEND"}

    def _mask(self, token):
        raw = token["raw"]
        for opening, marker in self.markers.items():
            raw = raw.replace(opening, marker)
        return dict(token, raw=raw)

    def block_code(self, token, state):
        return super().block_code(self._mask(token), state)

    def codespan(self, token, state):
        return super().codespan(self._mask(token), state)

    def text(self, token, state):
        return token["raw"] if "{{" in token["raw"] else super().text(token, state)

    def inline_html(self, token, state):
        return "" if token["raw"].startswith("<!--") else token["raw"]

    def block_html(self, token, state):
        raw = token["raw"]
        return str(html_soup(raw)) + "\n\n" if "<!--" in raw else raw + "\n\n"

    def restore(self, text):
        for opening, marker in self.markers.items():
            text = text.replace(marker, opening)
        return text


class DocumentRenderer(MarkdownRenderer):
    def __init__(self, rewrite):
        super().__init__()
        self.rewrite = rewrite

    def link(self, token, state):
        attrs = token["attrs"]
        target = self.rewrite(attrs["url"], self._label(token.get("children", [])), attrs.get("title", ""), token["type"] == "image")
        token = dict(token, attrs=dict(attrs, url=target))
        token.pop("label", None)
        return super().link(token, state)

    def text(self, token, state):
        return token["raw"] if "{{" in token["raw"] else super().text(token, state)

    def render_referrences(self, state):
        return []

    @classmethod
    def _label(cls, tokens):
        return "".join(token.get("raw", cls._label(token.get("children", []))) for token in tokens)

    def inline_html(self, token, state):
        raw = token["raw"]
        if raw.startswith("<!--"):
            return ""
        return self._html(raw) if raw.lower().startswith("<img") else raw

    def block_html(self, token, state):
        return self._html(token["raw"]) + "\n\n"

    def _html(self, raw):
        soup = html_soup(raw)
        for image in soup.find_all("img", src=True):
            attrs = {"url": image["src"], "title": image.get("title", "")}
            token = {"type": "image", "attrs": attrs, "children": [{"type": "text", "raw": image.get("alt", "")}]}
            image.replace_with(self.image(token, None))
        return str(soup)


def normalize_markdown(text, shortcodes, rewrite, warnings):
    plugins = ["table", "strikethrough", "task_lists"]
    protector = CodeMaskRenderer(text)
    protected = mistune.create_markdown(renderer=protector, plugins=plugins)(text)
    expanded = shortcodes.process(protected, warnings)
    expanded = protector.restore(expanded)
    return mistune.create_markdown(renderer=DocumentRenderer(rewrite), plugins=plugins)(expanded)
