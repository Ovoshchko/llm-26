from qdrant_client import models


class CollectionManager:
    def __init__(self, client, config):
        self.client = client
        self.config = config

    def prepare(self, name, bundle, reset=False):
        exists = self.client.collection_exists(name)
        if exists and reset:
            self.client.delete_collection(name)
            exists = False
        if exists:
            self._validate(name, bundle)
        else:
            self._create(name, bundle)
        self._create_payload_indexes(name)

    def _create(self, name, bundle):
        self.client.create_collection(
            collection_name=name,
            vectors_config={"dense": models.VectorParams(size=bundle.manifest["dimension"], distance=models.Distance.COSINE)},
            hnsw_config=models.HnswConfigDiff(m=self.config.m, ef_construct=self.config.ef_construction),
            metadata={"rag_index": self.config.identity(bundle)},
        )

    def _validate(self, name, bundle):
        info = self.client.get_collection(name)
        vectors = info.config.params.vectors
        compatible = (
            isinstance(vectors, dict) and set(vectors) == {"dense"}
            and vectors["dense"].size == bundle.manifest["dimension"]
            and vectors["dense"].distance == models.Distance.COSINE
            and info.config.hnsw_config.m == self.config.m
            and info.config.hnsw_config.ef_construct == self.config.ef_construction
            and (info.config.metadata or {}).get("rag_index") == self.config.identity(bundle)
        )
        if not compatible:
            raise ValueError(f"Collection {name!r} has a different schema, model, corpus, or HNSW configuration. Use a new name or --drop-and-reindex")

    def _create_payload_indexes(self, name):
        schema = self.client.get_collection(name).payload_schema
        for field in ("document_id", "section_id", "chunk_id"):
            if field not in schema:
                self.client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD, wait=True)
