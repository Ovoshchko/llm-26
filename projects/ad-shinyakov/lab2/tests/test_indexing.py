import hashlib
import json
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from artifacts import file_hash, read_json, write_json
from embedder import EmbeddingBundle, EmbeddingConfig, EmbeddingPipeline, SentenceEncoder
from vector_store import IndexConfig, VectorLoader, VectorSearch


class DeterministicEncoder:
    def __init__(self):
        self.config = EmbeddingConfig(model="test-encoder", revision="test-commit")
        self.calls = 0

    def resolve(self):
        return self.config

    def encode_documents(self, texts, batch_size):
        self.calls += 1
        vectors = np.array([list(hashlib.sha256(text.encode()).digest()[:8]) for text in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class IndexingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input = self.root / "chunks"
        self.input.mkdir()
        chunks = [
            {"id": "chunk-a", "text": "Service directs traffic to Pods.", "document_id": "doc", "section_id": "section"},
            {"id": "chunk-b", "text": "A Deployment manages ReplicaSets.", "document_id": "doc", "section_id": "other"},
        ]
        (self.input / "chunks.jsonl").write_text("".join(json.dumps(chunk) + "\n" for chunk in chunks))
        write_json(self.input / "manifest.json", {"chunks": 2, "config": {"chunk_size": 800, "overlap": 100}})
        self.encoder = DeterministicEncoder()
        self.pipeline = EmbeddingPipeline(self.encoder, self.root / "embeddings")
        self.bundle = self.pipeline.run(self.input, batch_size=1)
        self.client = QdrantClient(":memory:")
        warnings.filterwarnings("ignore", message="Payload indexes have no effect.*")

    def tearDown(self):
        self.client.close()
        self.temporary.cleanup()

    def test_embedding_cache_and_rebuild(self):
        calls = self.encoder.calls
        self.assertEqual(self.bundle, self.pipeline.run(self.input))
        self.assertEqual(calls, self.encoder.calls)
        self.assertEqual(self.bundle, self.pipeline.run(self.input, rebuild=True))
        self.assertGreater(self.encoder.calls, calls)

    def test_changed_corpus_or_model_creates_new_version(self):
        self.encoder.config = replace(self.encoder.config, revision="new-commit")
        self.assertNotEqual(self.bundle, self.pipeline.run(self.input))
        self.encoder.config = replace(self.encoder.config, revision="test-commit")
        chunks = self.input / "chunks.jsonl"
        chunks.write_text(chunks.read_text().replace("traffic", "requests"))
        self.assertNotEqual(self.bundle, self.pipeline.run(self.input))

    def test_duplicate_ids_rejected(self):
        path = self.input / "chunks.jsonl"
        path.write_text(path.read_text() * 2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.pipeline.run(self.input)

    def test_artifact_corruption_rejected(self):
        path = self.bundle / "vectors.npy"
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "checksum"):
            EmbeddingBundle(self.bundle)

    def test_nonfinite_vectors_rejected_even_with_updated_checksum(self):
        path = self.bundle / "vectors.npy"
        vectors = np.load(path)
        vectors[0, 0] = np.nan
        np.save(path, vectors)
        manifest = read_json(self.bundle / "manifest.json")
        manifest["files"]["vectors.npy"] = file_hash(path)
        write_json(self.bundle / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "NaN"):
            EmbeddingBundle(self.bundle)

    def test_load_twice_search_and_reset(self):
        loader = VectorLoader(self.client, IndexConfig(ef_construction=100))
        first = loader.run(self.bundle, batch_size=1)
        second = loader.run(self.bundle)
        name = first["collection"]
        self.assertEqual(name, second["collection"])
        self.assertEqual(self.client.count(name, exact=True).count, 2)
        bundle = EmbeddingBundle(self.bundle)
        hits = VectorSearch(self.client, name).query(bundle.vectors[0], limit=1)
        self.assertEqual(hits[0].payload["chunk_id"], "chunk-a")
        filtered = VectorSearch(self.client, name).query(bundle.vectors[0], limit=2, query_filter=models.Filter(must=[models.FieldCondition(key="section_id", match=models.MatchValue(value="other"))]))
        self.assertEqual([hit.payload["chunk_id"] for hit in filtered], ["chunk-b"])
        loader.run(self.bundle, reset=True)
        self.assertEqual(self.client.count(name, exact=True).count, 2)

    def test_configuration_mismatch_refuses_overwrite(self):
        VectorLoader(self.client, IndexConfig(ef_construction=100)).run(self.bundle, collection="explicit")
        with self.assertRaisesRegex(ValueError, "different"):
            VectorLoader(self.client, IndexConfig(m=24, ef_construction=100)).run(self.bundle, collection="explicit")
        self.assertEqual(self.client.count("explicit", exact=True).count, 2)
        VectorLoader(self.client, IndexConfig(m=24, ef_construction=100)).run(self.bundle, collection="explicit", reset=True)
        self.assertEqual(self.client.get_collection("explicit").config.metadata["rag_index"]["hnsw"]["m"], 24)

    def test_stale_points_require_reset(self):
        loader = VectorLoader(self.client, IndexConfig(ef_construction=100))
        name = loader.run(self.bundle)["collection"]
        self.client.upsert(name, [models.PointStruct(id=99, vector={"dense": [1.0] + [0.0] * 7})])
        with self.assertRaisesRegex(ValueError, "stale"):
            loader.run(self.bundle)
        loader.run(self.bundle, reset=True)
        self.assertEqual(self.client.count(name, exact=True).count, 2)

    def test_interrupted_upload_can_be_resumed(self):
        loader = VectorLoader(self.client, IndexConfig(ef_construction=100))
        bundle = EmbeddingBundle(self.bundle)
        name = loader.config.collection_name(bundle)
        loader.collections.prepare(name, bundle)
        self.client.upsert(name, loader._points(bundle, 0, 1), wait=True)
        loader.run(self.bundle)
        self.assertEqual(self.client.count(name, exact=True).count, 2)

    def test_encoder_refuses_truncation(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=True):
                return list(text)
        class Model:
            tokenizer = Tokenizer()
        encoder = SentenceEncoder(EmbeddingConfig(max_length=10))
        encoder.model = Model()
        with self.assertRaisesRegex(ValueError, "silent truncation"):
            encoder.encode_documents(["x" * 11], 1)


if __name__ == "__main__":
    unittest.main()
