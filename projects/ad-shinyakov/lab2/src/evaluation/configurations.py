from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter, sleep

import numpy as np
from qdrant_client import models

from artifacts import file_hash, fingerprint, read_json, write_json
from embedder import EmbeddingBundle, EmbeddingConfig, EmbeddingPipeline, SentenceEncoder
from vector_store import IndexConfig, VectorLoader
from .metrics import aggregate, score_ranking


def project_vectors(vectors, dimension):
    if dimension <= 0 or dimension > vectors.shape[1]:
        raise ValueError("Invalid output dimension")
    result = np.array(vectors[:, :dimension], dtype=np.float32, copy=True)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(result).all() or np.any(norms == 0):
        raise ValueError("Cannot normalize projected vectors")
    return result / norms


class CachedDimensionEncoder:
    def __init__(self, bundle, dimension):
        self.bundle = bundle
        self.config = replace(EmbeddingConfig(**bundle.identity["embedding"]), dimensions=dimension)
        if self.config.model != "Qwen/Qwen3-Embedding-0.6B":
            raise ValueError("Dimension projection is verified only for Qwen3-Embedding-0.6B")

    def resolve(self):
        return self.config

    def encode_documents(self, texts, batch_size):
        if texts != [c["text"] for c in self.bundle.chunks]:
            raise ValueError("Cached document vectors do not match input texts")
        return project_vectors(self.bundle.vectors, self.config.dimensions)


class ConfigurationBenchmark:
    DIMENSIONS = (1024, 512, 256)

    def __init__(self, client, embeddings, chunks, questions, labels):
        self.client = client
        self.source = EmbeddingBundle(embeddings)
        self.chunks_dir = chunks
        self.dataset = read_json(questions)
        self.labels = read_json(labels)
        self.labels_sha256 = file_hash(labels)
        self.chunk_manifest = read_json(chunks / "manifest.json")
        if self.source.manifest["dimension"] != 1024:
            raise ValueError("Expected full 1024-dimensional source vectors")
        expected_chunks = self.source.manifest["files"]["chunks.jsonl"]
        if expected_chunks != self.labels["chunks_sha256"] or expected_chunks != file_hash(chunks / "chunks.jsonl"):
            raise ValueError("Chunks, embeddings and labels must match")
        if file_hash(questions) != self.labels["questions_sha256"]:
            raise ValueError("Questions differ from gold labels")
        if file_hash(chunks / "manifest.json") != self.labels["manifest_sha256"]:
            raise ValueError("Chunk manifest differs from gold labels")
        if self.source.identity["chunk_manifest_sha256"] != self.labels["manifest_sha256"]:
            raise ValueError("Embedding chunk manifest differs from gold labels")
        self.gold = {q["question_id"]: q for q in self.labels["judgments"]}
        if set(self.gold) != {q["id"] for q in self.dataset["questions"]}:
            raise ValueError("Question IDs differ from gold labels")
        self.chunk_by_id = {c["id"]: c for c in self.source.chunks}

    def run(self, output, repetitions=10, device=None):
        if repetitions < 2:
            raise ValueError("At least two latency repetitions are required")
        output.mkdir(parents=True, exist_ok=True)
        config = EmbeddingConfig(**self.source.identity["embedding"])
        encoder = SentenceEncoder(config, device=device)
        started = perf_counter()
        encoder.load()
        load_seconds = perf_counter() - started
        started = perf_counter()
        queries = encoder.encode_queries([q["question"] for q in self.dataset["questions"]])
        encode_seconds = perf_counter() - started
        variants = [self.prepare(d, queries, output) for d in self.DIMENSIONS]
        latencies = {v["name"]: [] for v in variants}
        for variant in variants:
            for vector in variant["query_vectors"]:
                self.search(variant["collection"], vector)
        for repeat in range(repetitions):
            order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
            for variant in order:
                for vector in variant["query_vectors"]:
                    started = perf_counter()
                    self.search(variant["collection"], vector)
                    latencies[variant["name"]].append((perf_counter() - started) * 1000)
        for variant in variants:
            measured = latencies[variant["name"]]
            variant["latency"] = {
                "samples_ms": measured, "count": len(measured),
                "mean_ms": float(np.mean(measured)),
                "p50_ms": float(np.percentile(measured, 50)),
                "p95_ms": float(np.percentile(measured, 95)),
            }
            del variant["query_vectors"]
        baseline = variants[0]["metrics"]
        eligible = [v for v in variants if (
            v["metrics"]["all"]["recall@10"] >= baseline["all"]["recall@10"] - 0.02 - 1e-9
            and v["metrics"]["all"]["mrr@10"] >= baseline["all"]["mrr@10"] - 0.02 - 1e-9
            and v["metrics"]["multi_hop"]["complete_evidence@10"] >= baseline["multi_hop"]["complete_evidence@10"]
        )]
        selected = min(eligible, key=lambda v: v["cost"]["raw_vector_bytes"])
        report = {
            "experiment": "stage_3_4_embedding_dimensions",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "questions_sha256": self.labels["questions_sha256"],
            "labels_sha256": self.labels_sha256, "labels": self.labels,
            "source_embedding_manifest": self.source.manifest,
            "query_model_load_seconds": load_seconds,
            "query_forward_seconds_shared": encode_seconds,
            "device": str(encoder.model.device),
            "qdrant_version": self.client.info().version,
            "packages": {p: version(p) for p in ("numpy", "sentence-transformers", "transformers", "torch", "qdrant-client")},
            "selection_rule": "Smallest raw vector storage with Recall@10 and MRR@10 loss <= 0.02 vs 1024, and no loss of multi-hop complete_evidence@10",
            "recommended": selected["name"],
            "ks": [5, 10], "latency_repetitions": repetitions,
            "variants": variants,
        }
        write_json(output / "results.json", report)
        from .configuration_report import write_configuration_report

        write_configuration_report(report, output)
        return report

    def prepare(self, dimension, full_queries, output):
        started = perf_counter()
        cached = CachedDimensionEncoder(self.source, dimension)
        directory = EmbeddingPipeline(cached, self.source.directory.parent).run(self.chunks_dir)
        bundle = EmbeddingBundle(directory)
        query_vectors = project_vectors(full_queries, dimension)
        projection_seconds = perf_counter() - started
        np.save(output / f"queries_{dimension}.npy", query_vectors, allow_pickle=False)
        index = IndexConfig(m=16, ef_construction=128)
        collection = "eval_config_" + fingerprint({"index": index.identity(bundle), "run": output.name})[:20]
        reused = self.client.collection_exists(collection)
        started = perf_counter()
        VectorLoader(self.client, index).run(directory, collection=collection)
        self.client.update_collection(
            collection,
            hnsw_config=models.HnswConfigDiff(full_scan_threshold=10, on_disk=False),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1, default_segment_number=1),
        )
        self.wait_for_index(collection, len(bundle.chunks))
        setup_seconds = perf_counter() - started
        rows = [self.evaluate_question(q, vector, collection)
                for q, vector in zip(self.dataset["questions"], query_vectors)]
        exact_overlap = []
        for row, vector in zip(rows, query_vectors):
            exact = {h.payload["chunk_id"] for h in self.search(collection, vector, exact=True)}
            exact_overlap.append(len(exact.intersection(h["chunk_id"] for h in row["hits"])) / 10)
        info = self.client.get_collection(collection)
        result = {
            "name": f"dense_{dimension}", "collection": collection,
            "configuration": {
                "splitter": {k: self.chunk_manifest[k] for k in ("splitter", "config", "tokenizer")},
                "embedding": asdict(cached.config),
                "index": {"distance": "Cosine", "vector_name": "dense", "dimension": dimension,
                          "m": 16, "ef_construction": 128, "ef_search": 128,
                          "full_scan_threshold": 10, "indexing_threshold": 1,
                          "default_segment_number": 1, "quantization": None},
            },
            "embedding_manifest": bundle.manifest,
            "collection_info": info.model_dump(mode="json"),
            "derivation": {"method": "MRL: first d dimensions, then L2 normalization",
                           "source_embedding_version": self.source.version,
                           "source_vectors_sha256": self.source.manifest["files"]["vectors.npy"]},
            "cost": {"raw_vector_bytes": int(bundle.vectors.nbytes),
                     "vectors_npy_bytes": (directory / "vectors.npy").stat().st_size,
                     "projection_and_bundle_seconds": projection_seconds,
                     "collection_setup_seconds": setup_seconds, "collection_reused": reused},
            "query_vectors_sha256": file_hash(output / f"queries_{dimension}.npy"),
            "ann_vs_exact_overlap_at_10": float(np.mean(exact_overlap)),
            "metrics": {group: aggregate([r["metrics"] for r in rows if group == "all" or r["type"] == group])
                        for group in ("all", "single_hop", "multi_hop")},
            "questions": rows, "query_vectors": query_vectors,
        }
        write_json(output / f"configuration_{dimension}.json", {k: v for k, v in result.items() if k != "query_vectors"})
        print(f"Ready {result['name']}: indexed={info.indexed_vectors_count}, metrics={result['metrics']['all']}", flush=True)
        return result

    def wait_for_index(self, collection, count, timeout=180):
        started = perf_counter()
        while perf_counter() - started < timeout:
            info = self.client.get_collection(collection)
            if info.points_count == count and info.indexed_vectors_count == count and str(info.status) == "green":
                return
            sleep(1)
        raise RuntimeError(f"HNSW index is not ready: {collection}")

    def search(self, collection, vector, exact=False):
        return self.client.query_points(
            collection, query=vector.tolist(), using="dense", limit=10,
            search_params=models.SearchParams(hnsw_ef=128, exact=exact),
            with_payload=["chunk_id"],
        ).points

    def evaluate_question(self, question, vector, collection):
        hits = self.search(collection, vector)
        ids = [h.payload["chunk_id"] for h in hits]
        return {
            "question_id": question["id"], "type": question["type"], "question": question["question"],
            "metrics": score_ranking(ids, self.gold[question["id"]]),
            "hits": [{"rank": i, "chunk_id": h.payload["chunk_id"], "score": h.score,
                      **{key: self.chunk_by_id[h.payload["chunk_id"]][key]
                         for key in ("text", "heading_path", "source_path")}}
                     for i, h in enumerate(hits, 1)],
        }
