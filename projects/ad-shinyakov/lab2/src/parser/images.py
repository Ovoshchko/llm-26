import hashlib
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit


class ImageStore:
    def __init__(self, source, storage):
        self.source = source
        self.storage = storage

    def save(self, url, document, destination):
        parts = urlsplit(url)
        if parts.scheme or parts.netloc:
            return url, {"source_url": url, "external": True}
        raw = unquote(parts.path)
        if raw.startswith("/"):
            candidates = [self.source.root / "static" / raw.lstrip("/")]
        else:
            candidates = [document.parent / raw, self.source.root / "static" / raw]
        for candidate in candidates:
            path = self.source.resolve(candidate)
            if path.is_file():
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                target = self.storage.root / "assets" / (digest + path.suffix.lower())
                self.storage.write(target, data)
                local = Path(os.path.relpath(target, destination.parent)).as_posix()
                return local, {"source_path": path.relative_to(self.source.root).as_posix(), "path": target.relative_to(self.storage.root).as_posix(), "sha256": digest}
        raise ValueError(f"Image not found: {url}")
