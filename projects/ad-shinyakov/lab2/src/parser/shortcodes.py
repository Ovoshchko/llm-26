import re
import shlex


class ShortcodeProcessor:
    pattern = re.compile(r"{{[<%]\s*(.*?)\s*[>%]}}", re.DOTALL)

    def __init__(self, source):
        self.source = source

    def process(self, text, warnings):
        return self.pattern.sub(lambda match: self._expand(match, warnings), text)

    def _expand(self, match, warnings):
        tokens = shlex.split(match.group(1))
        if not tokens:
            return ""
        name = tokens[0]
        attrs = dict(token.split("=", 1) for token in tokens[1:] if "=" in token)
        if name in {"ref", "relref"}:
            target = attrs.get("path", tokens[1] if len(tokens) > 1 else "")
            target = target.removesuffix(".md").replace("/_index", "")
            return target
        if name == "figure":
            title = attrs.get("caption", attrs.get("title", ""))
            return f'![{attrs.get("alt", title)}]({attrs.get("src", "")})\n\n{title}'
        if name in {"code_sample", "codenew"}:
            return self._code_sample(attrs, warnings)
        if name in {"glossary_tooltip", "glossary_definition"}:
            return self._glossary(name, attrs, warnings)
        if name == "feature-state":
            return f'Feature state: {attrs.get("state", "")} ({attrs.get("for_k8s_version", "")})'
        if name in {"note", "warning", "caution"}:
            return f"\n\n**{name.capitalize()}:**\n\n"
        if name.startswith("/") or name in {"tab", "tabs", "capture"}:
            return ""
        warnings.append(f"Unsupported shortcode preserved: {name}")
        return match.group(0)

    def _code_sample(self, attrs, warnings):
        name = attrs.get("file", "")
        if not name:
            warnings.append("Code sample without file")
            return "[Code sample unavailable]"
        path = self.source.resolve(self.source.content / "examples" / name)
        if not path.is_file():
            warnings.append(f"Code sample not found: {name}")
            return f"[Code sample unavailable: {name}]"
        code = path.read_text(encoding="utf-8")
        fence = "`" * max(3, max((len(run) + 1 for run in re.findall(r"`+", code)), default=3))
        return f"\n\n{fence}{path.suffix.lstrip('.')}\n{code.rstrip()}\n{fence}\n\n"

    def _glossary(self, name, attrs, warnings):
        term = attrs.get("term_id", "")
        path = self.source.resolve(self.source.content / "docs" / "reference" / "glossary" / f"{term}.md")
        if not path.is_file():
            warnings.append(f"Glossary entry not found: {term}")
            return attrs.get("text", term)
        metadata, body = self.source.read(path)
        if name == "glossary_tooltip":
            return attrs.get("text", metadata.get("title", term))
        definition = metadata.get("short_description") or body
        return attrs.get("prepend", "") + " " + str(definition).strip()
