from qdrant_client import models


class VectorSearch:
    def __init__(self, client, collection, ef_search=128):
        if ef_search <= 0:
            raise ValueError("ef_search must be positive")
        self.client = client
        self.collection = collection
        self.ef_search = ef_search

    def query(self, vector, limit=5, query_filter=None):
        if limit <= 0:
            raise ValueError("limit must be positive")
        return self.client.query_points(
            collection_name=self.collection, query=vector.tolist() if hasattr(vector, "tolist") else vector,
            using="dense", limit=limit, query_filter=query_filter,
            search_params=models.SearchParams(hnsw_ef=max(self.ef_search, limit)),
            with_payload=True,
        ).points
