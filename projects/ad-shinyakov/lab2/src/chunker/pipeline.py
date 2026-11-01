import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from parser.source import RepositorySource
from parser.storage import FileStorage


class ChunkingPipeline:
    def __init__(self, chunker, output):
        self.chunker = chunker
        self.storage = FileStorage(Path(output).resolve())

    def run(self, source):
        source = Path(source).resolve()
        documents = source / "documents" if (source / "documents").is_dir() else source
        paths = sorted(documents.rglob("*.md"))
        if not paths:
            raise ValueError(f"No Markdown files found: {documents}")
        if self.storage.root.is_relative_to(documents):
            raise ValueError("Chunk output must be outside the input documents directory")
        chunks, inputs = [], []
        for path in paths:
            metadata, text = RepositorySource.read(path)
            sidecar = path.with_suffix(".json")
            if sidecar.is_file():
                metadata.update(json.loads(sidecar.read_text()))
            relative = path.relative_to(documents).as_posix()
            document_id = self.chunker._id(metadata.get("source_path", relative), metadata.get("source_commit") or "")
            document_chunks = self.chunker.split(text, document_id, metadata)
            for chunk in document_chunks:
                chunk["document_path"] = relative
            chunks.extend(document_chunks)
            inputs.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "metadata_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest() if sidecar.is_file() else None})
        manifest = {"splitter": f"{self.chunker.config.splitter}-library-v8", "config": asdict(self.chunker.config), "tokenizer": self.chunker.tokenizer.model, "inputs": inputs, "documents": len(paths), "chunks": len(chunks)}
        version = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
        destination = self.storage.root / version
        data = "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)
        self.storage.write(destination / "chunks.jsonl", data.encode())
        self.storage.write(destination / "manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode())
        return destination, len(chunks)
