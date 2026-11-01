import uuid
from dataclasses import asdict
from datetime import datetime, timezone

from qdrant_client import models

from artifacts import fingerprint, write_json
from embedder.bundle import EmbeddingBundle
from .collection import CollectionManager
from .config import IndexConfig


class VectorLoader:
    def __init__(self, client, config=None):
        self.client = client
        self.config = config or IndexConfig()
        self.collections = CollectionManager(client, self.config)

    def run(self, source, collection=None, batch_size=128, reset=False, ef_search=128):
        if batch_size <= 0 or ef_search <= 0:
            raise ValueError("batch_size and ef_search must be positive")
        bundle = EmbeddingBundle(source)
        name = collection or self.config.collection_name(bundle)
        self.collections.prepare(name, bundle, reset)
        for start in range(0, len(bundle.chunks), batch_size):
            self.client.upsert(name, self._points(bundle, start, batch_size), wait=True)
            print(f"Uploaded {min(start + batch_size, len(bundle.chunks))}/{len(bundle.chunks)} to {name}")
        self._verify(name, bundle)
        manifest = self._manifest(name, bundle, ef_search)
        write_json(bundle.directory / f"upload_manifest_{fingerprint(name)[:12]}.json", manifest)
        return manifest

    @staticmethod
    def point_id(chunk_id):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "kubernetes-rag:chunk:" + chunk_id))

    def _points(self, bundle, start, batch_size):
        points = []
        for index in range(start, min(start + batch_size, len(bundle.chunks))):
            chunk = bundle.chunks[index]
            payload = dict(chunk, chunk_id=chunk["id"], embedding_version=bundle.version)
            points.append(models.PointStruct(
                id=self.point_id(chunk["id"]),
                vector={"dense": bundle.vectors[index].tolist()},
                payload=payload,
            ))
        return points

    def _verify(self, name, bundle):
        count = self.client.count(name, exact=True).count
        if count != len(bundle.chunks):
            raise ValueError(f"Collection contains {count} points, expected {len(bundle.chunks)}; use --drop-and-reindex to remove stale points")

    def _manifest(self, name, bundle, ef_search):
        info = self.client.get_collection(name)
        return {
            "collection": name, "embedding_version": bundle.version,
            "dimension": bundle.manifest["dimension"], "count": len(bundle.chunks),
            "distance": "Cosine", "vector_name": "dense",
            "hnsw": asdict(self.config), "search": {"hnsw_ef": ef_search},
            "indexed_vectors_count": info.indexed_vectors_count,
            "collection_status": str(info.status),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
