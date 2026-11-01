from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

import numpy as np

from artifacts import file_hash, read_json, write_json
from embedder import EmbeddingBundle, EmbeddingConfig, SentenceEncoder
from vector_store import IndexConfig
from vector_store.collection import CollectionManager
from .metrics import aggregate, score_ranking
from .report import write_report
from .retrievers import BM25Retriever, DenseRetriever, reciprocal_rank_fusion


class RetrievalEvaluation:
    def __init__(self, client, embeddings, questions, labels, collection=None):
        self.client = client
        self.bundle = EmbeddingBundle(embeddings)
        self.chunks = {c["id"]: c for c in self.bundle.chunks}
        self.dataset = read_json(questions)
        self.labels = read_json(labels)
        self.labels_sha256 = file_hash(labels)
        if file_hash(questions) != self.labels["questions_sha256"]:
            raise ValueError("Questions differ from annotated questions")
        if self.bundle.manifest["files"]["chunks.jsonl"] != self.labels["chunks_sha256"]:
            raise ValueError("Embeddings and gold labels refer to different chunks")
        if self.bundle.identity["chunk_manifest_sha256"] != self.labels["manifest_sha256"]:
            raise ValueError("Chunk manifest differs from gold labels")
        self.gold = {q["question_id"]: q for q in self.labels["judgments"]}
        if set(self.gold) != {q["id"] for q in self.dataset["questions"]}:
            raise ValueError("Question and judgment IDs differ")
        known = {c["id"] for c in self.bundle.chunks}
        if any(not set(q["relevant_chunk_ids"]) <= known for q in self.gold.values()):
            raise ValueError("Gold labels reference missing chunks")
        self.index = IndexConfig()
        self.collection = collection or self.index.collection_name(self.bundle)
        CollectionManager(client, self.index)._validate(self.collection, self.bundle)
        if client.count(self.collection, exact=True).count != len(self.bundle.chunks):
            raise ValueError("Collection has an unexpected point count")

    def run(self, output, device=None):
        output.mkdir(parents=True, exist_ok=True)
        config = EmbeddingConfig(**self.bundle.identity["embedding"])
        encoder = SentenceEncoder(config, device=device)
        started = perf_counter()
        encoder.load()
        load_seconds = perf_counter() - started
        started = perf_counter()
        vectors = encoder.encode_queries([q["question"] for q in self.dataset["questions"]])
        encode_seconds = perf_counter() - started
        np.save(output / "queries.npy", vectors, allow_pickle=False)
        started = perf_counter()
        lexical = BM25Retriever(self.bundle.chunks)
        bm25_build_seconds = perf_counter() - started
        dense = DenseRetriever(self.client, self.collection)
        rows = {name: [] for name in ("dense", "bm25", "hybrid_rrf")}
        for question, vector in zip(self.dataset["questions"], vectors):
            started = perf_counter()
            dense_hits = dense.search(vector)
            dense_seconds = perf_counter() - started
            started = perf_counter()
            bm25_hits = lexical.search(question["question"])
            bm25_seconds = perf_counter() - started
            started = perf_counter()
            fused = reciprocal_rank_fusion([dense_hits, bm25_hits])
            fusion_seconds = perf_counter() - started
            rows["dense"].append(self.score(question, dense_hits[:10], dense_seconds))
            rows["bm25"].append(self.score(question, bm25_hits[:10], bm25_seconds))
            row = self.score(question, fused, dense_seconds + bm25_seconds + fusion_seconds)
            row["candidate_lists"] = {"dense": dense_hits, "bm25": bm25_hits}
            rows["hybrid_rrf"].append(row)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_manifest": self.bundle.manifest,
            "labels_sha256": self.labels_sha256,
            "questions_sha256": self.labels["questions_sha256"],
            "labels": self.labels,
            "collection": self.collection,
            "collection_info": self.client.get_collection(self.collection).model_dump(mode="json"),
            "qdrant_version": self.client.info().version,
            "packages": {p: version(p) for p in ("sentence-transformers", "transformers", "torch", "qdrant-client", "rank-bm25")},
            "device": str(encoder.model.device),
            "model_load_seconds": load_seconds,
            "bm25_build_seconds": bm25_build_seconds,
            "query_vectors_sha256": file_hash(output / "queries.npy"),
            "experiment": "dense_vs_bm25_vs_hybrid_rrf",
            "ks": [5, 10],
            "mrr_definition": "Mean reciprocal rank truncated at k; no hit within k gives zero",
            "precision_caveat": "Unlisted chunks are unjudged, so precision counts known positives only",
            "variants": [],
        }
        settings = {
            "dense": {"hnsw_ef": 128, "exact": False, "candidate_limit": 100, "limit": 10},
            "bm25": {"algorithm": "BM25Okapi", "k1": 1.5, "b": 0.75, "epsilon": 0.25,
                     "tokenizer": "Unicode alphanumeric tokens, casefold, no stemming/translation/stopwords",
                     "positive_scores_only": True, "candidate_limit": 100, "limit": 10},
            "hybrid_rrf": {"inputs": ["dense", "bm25"], "constant": 60, "weights": [1, 1],
                           "candidates_per_input": 100, "limit": 10},
        }
        for name, question_rows in rows.items():
            result = {
                "name": name, "query_prefix": config.query_prefix if name != "bm25" else None,
                "search": settings[name],
                "query_encode_seconds": encode_seconds if name != "bm25" else 0.0,
                "search_seconds": sum(row["search_seconds"] for row in question_rows),
                "metrics": {
                    group: aggregate([r["metrics"] for r in question_rows if group == "all" or r["type"] == group])
                    for group in ("all", "single_hop", "multi_hop")
                },
                "questions": question_rows,
            }
            report["variants"].append(result)
        write_json(output / "results.json", report)
        write_report(report, output)
        return report

    def score(self, question, hits, elapsed):
        ranking = [hit["chunk_id"] for hit in hits]
        return {
            "question_id": question["id"], "type": question["type"],
            "question": question["question"], "search_seconds": elapsed,
            "metrics": score_ranking(ranking, self.gold[question["id"]]),
            "hits": [{"rank": i, **hit, **{key: self.chunks[hit["chunk_id"]][key]
                      for key in ("source_path", "heading_path", "text")}}
                     for i, hit in enumerate(hits, 1)],
        }
