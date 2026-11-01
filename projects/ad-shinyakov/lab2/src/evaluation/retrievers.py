from collections import defaultdict
import re

import numpy as np
from qdrant_client import models
from rank_bm25 import BM25Okapi


class DenseRetriever:
    def __init__(self, client, collection):
        self.client = client
        self.collection = collection

    def search(self, vector, limit=100):
        hits = self.client.query_points(
            self.collection, query=vector.tolist(), using="dense", limit=limit,
            search_params=models.SearchParams(hnsw_ef=max(128, limit)),
            with_payload=["chunk_id"],
        ).points
        return [{"chunk_id": h.payload["chunk_id"], "score": h.score} for h in hits]


class BM25Retriever:
    def __init__(self, chunks):
        self.ids = [c["id"] for c in chunks]
        self.index = BM25Okapi([self.tokenize(c["text"]) for c in chunks], k1=1.5, b=0.75, epsilon=0.25)

    @staticmethod
    def tokenize(text):
        return re.findall(r"[^\W_]+", text.casefold())

    def search(self, question, limit=100):
        scores = self.index.get_scores(self.tokenize(question))
        order = np.argsort(-scores, kind="stable")
        return [{"chunk_id": self.ids[i], "score": float(scores[i])} for i in order[:limit] if scores[i] > 0]


def reciprocal_rank_fusion(rankings, limit=10, constant=60):
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    scores = defaultdict(float)
    for ranking in rankings:
        ids = [hit["chunk_id"] for hit in ranking]
        if len(ids) != len(set(ids)):
            raise ValueError("RRF input contains duplicate chunk IDs")
        for rank, cid in enumerate(ids, 1):
            scores[cid] += 1 / (constant + rank)
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:limit]
    return [{"chunk_id": cid, "score": scores[cid]} for cid in ordered]
