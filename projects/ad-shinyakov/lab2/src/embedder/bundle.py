from pathlib import Path

import numpy as np

from artifacts import file_hash, fingerprint, read_chunks, read_json


class EmbeddingBundle:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = read_json(self.directory / "manifest.json")
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported embedding artifact schema")
        self.identity = self.manifest["identity"]
        self.version = fingerprint(self.identity)
        if self.manifest.get("version") != self.version:
            raise ValueError("Embedding manifest identity mismatch")
        for name in ("chunks.jsonl", "vectors.npy"):
            if file_hash(self.directory / name) != self.manifest["files"][name]:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        if self.identity["chunks_sha256"] != self.manifest["files"]["chunks.jsonl"]:
            raise ValueError("Chunks do not match embedding identity")
        self.chunks = read_chunks(self.directory / "chunks.jsonl")
        self.vectors = np.load(self.directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
        self._validate_vectors()

    def _validate_vectors(self):
        expected = (len(self.chunks), self.manifest["dimension"])
        if self.vectors.ndim != 2 or self.vectors.shape != expected or len(self.chunks) != self.manifest["count"]:
            raise ValueError(f"Vector shape/count mismatch: expected {expected}, got {self.vectors.shape}")
        if self.vectors.dtype != np.float32:
            raise ValueError("Expected float32 vectors")
        for start in range(0, len(self.chunks), 1024):
            batch = self.vectors[start:start + 1024]
            norms = np.linalg.norm(batch, axis=1)
            if not np.isfinite(batch).all() or np.any(norms == 0):
                raise ValueError("Vectors contain NaN, Inf, or zero vectors")
            if not np.allclose(norms, 1.0, atol=1e-4):
                raise ValueError("Expected normalized vectors")
