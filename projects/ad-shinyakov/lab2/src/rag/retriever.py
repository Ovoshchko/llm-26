from embedder import EmbeddingBundle, EmbeddingConfig, SentenceEncoder
from vector_store import IndexConfig, VectorSearch
from vector_store.collection import CollectionManager


class DocumentRetriever:
    def __init__(self, client, embeddings, collection=None, ef_search=128, min_score=0.5, device=None):
        if min_score is not None and not -1 <= min_score <= 1:
            raise ValueError("min_score must be between -1 and 1")
        bundle = EmbeddingBundle(embeddings)
        index = IndexConfig()
        self.collection = collection or index.collection_name(bundle)
        CollectionManager(client, index)._validate(self.collection, bundle)
        if client.count(self.collection, exact=True).count != len(bundle.chunks):
            raise ValueError("Collection point count differs from embeddings")
        self.encoder = SentenceEncoder(EmbeddingConfig(**bundle.identity["embedding"]), device=device)
        self.search = VectorSearch(client, self.collection, ef_search)
        self.chunks = {c["id"]: c for c in bundle.chunks}
        self.embedding_version = bundle.version
        self.min_score = min_score

    def retrieve(self, question, top_k):
        vector = self.encoder.encode_query(question)
        contexts = []
        for hit in self.search.query(vector, limit=top_k):
            if self.min_score is not None and hit.score < self.min_score:
                continue
            chunk_id = hit.payload["chunk_id"]
            if chunk_id not in self.chunks:
                raise RuntimeError("Qdrant returned a chunk outside the embedding bundle")
            chunk = self.chunks[chunk_id]
            contexts.append({
                "chunk_id": chunk_id, "score": hit.score, "text": chunk["text"],
                "source_url": chunk["source_url"], "source_path": chunk["source_path"],
                "section": " > ".join(chunk["heading_path"]),
            })
        return contexts
