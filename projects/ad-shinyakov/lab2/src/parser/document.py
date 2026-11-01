import json
import re
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

import frontmatter

from .shortcodes import ShortcodeProcessor
from .markdown import normalize_markdown


class DocumentParser:
    def __init__(self, source, storage, images):
        self.source = source
        self.storage = storage
        self.images = images
        self.shortcodes = ShortcodeProcessor(source)

    def parse(self, path):
        path = self.source.resolve(path)
        original, body = self.source.read(path)
        destination = self.storage.root / "documents" / path.relative_to(self.source.content)
        metadata = self._metadata(path, original, destination)
        body = self._normalize(body, path, destination, metadata)
        title = metadata["title"]
        if not re.match(r"\s*#\s+", body):
            body = f"# {title}\n\n" + body.lstrip()
        self._save(destination, body, metadata)
        return metadata

    def _metadata(self, path, original, destination):
        return {
            "title": original.get("title", path.stem),
            "source_url": self.source.url(path),
            "source_path": path.relative_to(self.source.root).as_posix(),
            "source_commit": self.source.commit,
            "source_dirty": self.source.dirty,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "path": destination.relative_to(self.storage.root).as_posix(),
            "front_matter": original,
            "links": [], "images": [], "warnings": [],
        }

    def _normalize(self, body, path, destination, metadata):
        return normalize_markdown(
            body, self.shortcodes,
            lambda url, label, title, image: self._rewrite_link(url, label, title, image, path, destination, metadata),
            metadata["warnings"],
        )

    def _rewrite_link(self, url, label, title, image, path, destination, metadata):
        if not image:
            target = self._absolute_link(url, path)
            metadata["links"].append({"url": target, "text": label})
            return target
        record = {"original_url": url, "alt": label, "caption": title or ""}
        target = url
        try:
            target, saved = self.images.save(url, path, destination)
            record.update(saved)
        except ValueError as error:
            record["error"] = str(error)
            metadata["warnings"].append(str(error))
        metadata["images"].append(record)
        return target

    def _absolute_link(self, target, path):
        if target.startswith("#"):
            return self.source.url(path) + target
        if target.startswith("/"):
            return urljoin(self.source.base_url, target)
        if urlsplit(target).scheme or target.startswith("//"):
            return urljoin(self.source.base_url, target)
        location, separator, fragment = target.partition("#")
        candidate = (path.parent / location).resolve()
        if location and candidate.is_relative_to(self.source.content) and candidate.is_file() and candidate.suffix == ".md":
            return self.source.url(candidate) + (separator + fragment if separator else "")
        return urljoin(self.source.url(path), target)

    def _save(self, destination, body, metadata):
        keys = ("title", "source_url", "source_path", "source_commit", "source_dirty", "retrieved_at")
        post = frontmatter.Post(body, **{key: metadata[key] for key in keys})
        self.storage.write(destination, (frontmatter.dumps(post) + "\n").encode())
        data = json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n"
        self.storage.write(destination.with_suffix(".json"), data.encode())
