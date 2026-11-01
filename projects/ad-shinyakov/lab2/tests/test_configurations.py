from dataclasses import asdict
from types import SimpleNamespace
import unittest

import numpy as np

from embedder import EmbeddingConfig
from evaluation.configurations import CachedDimensionEncoder, project_vectors


class ConfigurationTests(unittest.TestCase):
    def test_mrl_projects_prefix_and_renormalizes(self):
        source = np.array([[3, 4, 12], [0, 5, 12]], dtype=np.float32) / 13
        original = source.copy()
        projected = project_vectors(source, 2)
        np.testing.assert_allclose(projected, [[0.6, 0.8], [0, 1]], atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(source, original)
        self.assertEqual(projected.dtype, np.float32)

    def test_invalid_dimensions_and_zero_prefix_rejected(self):
        for dimension in (0, 4):
            with self.assertRaises(ValueError):
                project_vectors(np.ones((2, 3)), dimension)
        with self.assertRaises(ValueError):
            project_vectors(np.array([[0, 0, 1]], dtype=np.float32), 2)

    def test_cached_encoder_checks_text_order(self):
        bundle = SimpleNamespace(
            identity={"embedding": asdict(EmbeddingConfig(revision="pinned"))},
            chunks=[{"text": "first"}, {"text": "second"}],
            vectors=np.array([[3, 4, 12], [0, 5, 12]], dtype=np.float32) / 13,
        )
        encoder = CachedDimensionEncoder(bundle, 2)
        self.assertEqual(encoder.resolve().dimensions, 2)
        self.assertEqual(encoder.resolve().revision, "pinned")
        self.assertEqual(encoder.encode_documents(["first", "second"], 2).shape, (2, 2))
        with self.assertRaises(ValueError):
            encoder.encode_documents(["second", "first"], 2)

    def test_unknown_model_projection_rejected(self):
        bundle = SimpleNamespace(identity={"embedding": asdict(EmbeddingConfig(model="other-model"))})
        with self.assertRaises(ValueError):
            CachedDimensionEncoder(bundle, 2)


if __name__ == "__main__":
    unittest.main()
