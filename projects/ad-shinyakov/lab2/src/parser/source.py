import subprocess
from pathlib import Path
from urllib.parse import urljoin

import frontmatter


class RepositorySource:
    def __init__(self, root, language="en", base_url="https://kubernetes.io"):
        self.root = Path(root).resolve()
        self.content = self.root / "content" / language
        if not (self.content / "docs").is_dir():
            raise ValueError(f"Documentation directory not found: {self.content / 'docs'}")
        self.base_url = base_url.rstrip("/") + "/"
        self.commit = self._git("rev-parse", "HEAD")
        self.dirty = bool(self._git("status", "--porcelain"))

    def _git(self, *args):
        result = subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    def resolve(self, path):
        candidate = Path(path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"Path escapes repository: {path}")
        return candidate

    def documents(self, sections):
        paths = set()
        for section in sections:
            path = self.resolve(self.content / section)
            if not path.exists():
                raise ValueError(f"Section not found: {section}")
            paths.update(path.rglob("*.md") if path.is_dir() else [path])
        return sorted(paths)

    def url(self, path):
        relative = Path(path).relative_to(self.content).with_suffix("").as_posix()
        if relative.endswith("/_index"):
            relative = relative[:-7]
        return urljoin(self.base_url, relative.rstrip("/") + "/")

    @staticmethod
    def read(path):
        text = Path(path).read_text(encoding="utf-8-sig")
        return frontmatter.parse(text)
