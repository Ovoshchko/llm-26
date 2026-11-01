import json
import sys

import yaml


class ParsingPipeline:
    def __init__(self, source, documents, storage):
        self.source = source
        self.documents = documents
        self.storage = storage

    def run(self, sections, limit=None):
        paths = self.source.documents(sections)
        if limit is not None:
            paths = paths[:limit]
        documents, errors = [], []
        for path in paths:
            try:
                document = self.documents.parse(path)
                documents.append(document)
                print(document["path"])
            except (OSError, ValueError, yaml.YAMLError) as error:
                errors.append({"path": str(path), "error": str(error)})
                print(f"Document error: {path}: {error}", file=sys.stderr)
        self._save_manifest(sections, documents, errors)
        return bool(errors or any(document["warnings"] for document in documents))

    def _save_manifest(self, sections, documents, errors):
        manifest = {"repository": str(self.source.root), "commit": self.source.commit, "dirty": self.source.dirty, "sections": sections, "documents": documents, "errors": errors}
        data = json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n"
        self.storage.write(self.storage.root / "manifest.json", data.encode())
