import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from artifacts import file_hash, fingerprint, read_chunks, read_json, write_json
from .bundle import EmbeddingBundle


class EmbeddingPipeline:
    def __init__(self, encoder, output):
        self.encoder = encoder
        self.output = Path(output).resolve()

    def run(self, source, batch_size=8, rebuild=False):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        source = Path(source).resolve()
        chunks_path = source / "chunks.jsonl" if source.is_dir() else source
        config = self.encoder.resolve()
        chunk_manifest = chunks_path.parent / "manifest.json"
        identity = {
            "schema": "dense-embeddings-v2", "embedding": asdict(config),
            "normalized": True, "dtype": "float32",
            "chunks_sha256": file_hash(chunks_path),
            "chunk_manifest_sha256": file_hash(chunk_manifest) if chunk_manifest.is_file() else None,
        }
        version = fingerprint(identity)
        destination = self.output / version[:16]
        if destination.exists() and not rebuild:
            EmbeddingBundle(destination)
            return destination
        self.output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="embedding-", dir=self.output) as temporary:
            staging = Path(temporary)
            shutil.copyfile(chunks_path, staging / "chunks.jsonl")
            if file_hash(staging / "chunks.jsonl") != identity["chunks_sha256"]:
                raise ValueError("Chunks changed during embedding preparation")
            chunks = read_chunks(staging / "chunks.jsonl")
            dimension = self._encode(chunks, staging / "vectors.npy", batch_size)
            manifest = {
                "schema_version": 1, "version": version, "identity": identity,
                "dimension": dimension, "count": len(chunks),
                "chunking": read_json(chunk_manifest).get("config") if chunk_manifest.is_file() else None,
                "files": {name: file_hash(staging / name) for name in ("vectors.npy", "chunks.jsonl")},
            }
            write_json(staging / "manifest.json", manifest)
            EmbeddingBundle(staging)
            if destination.exists():
                shutil.rmtree(destination)
            staging.rename(destination)
        return destination

    def _encode(self, chunks, destination, batch_size):
        vectors = np.asarray(self.encoder.encode_documents(
            [chunk["text"] for chunk in chunks], batch_size,
        ), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] == 0:
            raise ValueError("Encoder returned an inconsistent vector shape")
        np.save(destination, vectors, allow_pickle=False)
        return vectors.shape[1]
