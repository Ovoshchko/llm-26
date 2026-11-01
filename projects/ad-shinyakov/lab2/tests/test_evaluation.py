import unittest

from evaluation.metrics import aggregate, score_ranking
from evaluation.retrievers import BM25Retriever, reciprocal_rank_fusion


def judgment(*groups):
    return {
        "relevant_chunk_ids": sorted({cid for group in groups for cid in group}),
        "evidence_groups": [{"matches": [{"chunk_id": cid} for cid in group]} for group in groups],
    }


class EvaluationTests(unittest.TestCase):
    def test_known_ranking(self):
        result = score_ranking(["wrong", "a", "wrong2", "b", "wrong3"], judgment(["a"], ["b", "c"]), (5,))
        self.assertAlmostEqual(result["recall@5"], 2 / 3)
        self.assertEqual(result["precision@5"], 2 / 5)
        self.assertEqual(result["mrr@5"], 1 / 2)
        self.assertEqual(result["evidence_coverage@5"], 1)
        self.assertEqual(result["complete_evidence@5"], 1)

    def test_single_multi_hop_hit_is_incomplete(self):
        result = score_ranking(["a"], judgment(["a"], ["b"]), (5,))
        self.assertEqual(result["mrr@5"], 1)
        self.assertEqual(result["evidence_coverage@5"], 0.5)
        self.assertEqual(result["complete_evidence@5"], 0)
        self.assertEqual(result["precision@5"], 0.2)

    def test_multiple_evidence_groups_can_share_a_chunk(self):
        result = score_ranking(["a"], judgment(["a"], ["a", "b"]), (5,))
        self.assertEqual(result["complete_evidence@5"], 1)

    def test_mrr_is_truncated_at_k(self):
        result = score_ranking(["x", "y", "a"], judgment(["a"]), (2, 3))
        self.assertEqual(result["mrr@2"], 0)
        self.assertEqual(result["mrr@3"], 1 / 3)

    def test_rejects_duplicate_ranks_and_empty_gold(self):
        with self.assertRaises(ValueError):
            score_ranking(["a", "a"], judgment(["a"]))
        with self.assertRaises(ValueError):
            score_ranking(["a"], judgment())

    def test_macro_average(self):
        self.assertEqual(aggregate([{"recall@5": 1}, {"recall@5": 0}]), {"recall@5": 0.5})

    def test_bm25_finds_api_term_without_heading_search(self):
        chunks = [
            {"id": "service", "text": "Service targetPort 9376", "heading_path": ["Deployment"]},
            {"id": "deploy", "text": "Deployment replica rollout"},
            {"id": "volume", "text": "PersistentVolume storage claim"},
        ]
        retriever = BM25Retriever(chunks)
        self.assertEqual(retriever.search("Что означает TARGETPORT?")[0]["chunk_id"], "service")
        self.assertEqual(retriever.search("unmatchedword"), [])

    def test_bm25_tokenization_is_identical_for_queries_and_docs(self):
        self.assertEqual(BM25Retriever.tokenize("CPU targetPort node-zone Пример"),
                         ["cpu", "targetport", "node", "zone", "пример"])

    def test_rrf_promotes_shared_candidate_and_ignores_score_scales(self):
        dense = [{"chunk_id": "a", "score": 0.9}, {"chunk_id": "b", "score": 0.8}]
        bm25 = [{"chunk_id": "c", "score": 100}, {"chunk_id": "b", "score": 10}]
        result = reciprocal_rank_fusion([dense, bm25])
        self.assertEqual(result[0]["chunk_id"], "b")
        self.assertAlmostEqual(result[0]["score"], 2 / 62)
        self.assertEqual(len({hit["chunk_id"] for hit in result}), 3)

    def test_rrf_handles_empty_lexical_results(self):
        dense = [{"chunk_id": "b", "score": 0.9}, {"chunk_id": "a", "score": 0.8}]
        result = reciprocal_rank_fusion([dense, []], limit=1)
        self.assertEqual(result[0]["chunk_id"], "b")

    def test_rrf_rejects_duplicate_input(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[{"chunk_id": "a"}, {"chunk_id": "a"}]])


if __name__ == "__main__":
    unittest.main()
