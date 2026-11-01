import os
from contextlib import closing
import tempfile
import unittest
import uuid
from pathlib import Path

from qdrant_client import QdrantClient

from embedder import EmbeddingBundle, EmbeddingPipeline
from vector_store import IndexConfig, VectorLoader, VectorSearch
from test_indexing import DeterministicEncoder


@unittest.skipUnless(os.environ.get("QDRANT_TEST_URL"), "Set QDRANT_TEST_URL to test actual server HNSW configuration")
class QdrantServerTests(unittest.TestCase):
    def test_upload_search_reindex_and_resume(self):
        name = "rag_test_" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as temporary, closing(QdrantClient(url=os.environ["QDRANT_TEST_URL"])) as client:
            root = Path(temporary)
            (root / "chunks.jsonl").write_text('{"id":"a","text":"Service exposes Pods","document_id":"d","section_id":"s"}\n{"id":"b","text":"Deployment manages ReplicaSets","document_id":"d","section_id":"t"}\n')
            bundle_path = EmbeddingPipeline(DeterministicEncoder(), root / "embeddings").run(root / "chunks.jsonl")
            bundle = EmbeddingBundle(bundle_path)
            loader = VectorLoader(client)
            try:
                result = loader.run(bundle_path, collection=name, batch_size=1)
                self.assertEqual(result["count"], 2)
                info = client.get_collection(name)
                self.assertEqual(info.config.hnsw_config.m, 16)
                self.assertEqual(info.config.hnsw_config.ef_construct, 128)
                self.assertIn("document_id", info.payload_schema)
                loader.run(bundle_path, collection=name)
                self.assertEqual(client.count(name, exact=True).count, 2)
                hits = VectorSearch(client, name).query(bundle.vectors[0], limit=1)
                self.assertEqual(hits[0].payload["chunk_id"], "a")
                with self.assertRaisesRegex(ValueError, "different"):
                    VectorLoader(client, IndexConfig(m=24)).run(bundle_path, collection=name)
                changed = VectorLoader(client, IndexConfig(m=24))
                changed.run(bundle_path, collection=name, reset=True)
                self.assertEqual(client.get_collection(name).config.hnsw_config.m, 24)
                client.delete(name, [loader.point_id("b")], wait=True)
                changed.run(bundle_path, collection=name)
                self.assertEqual(client.count(name, exact=True).count, 2)
            finally:
                if client.collection_exists(name):
                    client.delete_collection(name)
