from dataclasses import asdict, dataclass

from artifacts import fingerprint


@dataclass(frozen=True)
class IndexConfig:
    m: int = 16
    ef_construction: int = 128

    def __post_init__(self):
        if self.m <= 0 or self.ef_construction < self.m:
            raise ValueError("m must be positive and ef_construction must be >= m")

    def identity(self, bundle):
        return {
            "schema": "qdrant-dense-v1", "embedding_version": bundle.version,
            "files": bundle.manifest["files"], "dimension": bundle.manifest["dimension"],
            "distance": "Cosine", "vector_name": "dense", "hnsw": asdict(self),
        }

    def collection_name(self, bundle):
        return "kubernetes_" + fingerprint(self.identity(bundle))[:20]
