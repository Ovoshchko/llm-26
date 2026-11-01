from contextlib import contextmanager
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx

from rag.api import create_app
from rag.evidence import EvidenceLedger
from rag.llm import GenerationConfig, OllamaGenerator
from rag.pipeline import RagPipeline, resolve_citations
from rag.prompt import PromptBuilder
from rag.retriever import DocumentRetriever
from rag.schemas import DraftAnswer, EvidenceQuote, EvidenceReview, SearchPlan, SearchStep
from rag.tokenizer import ChatTokenCounter


TEXT = "The Service forwards requests to TCP port 9376. The service port is 80."
CONTEXT = {
    "chunk_id": "chunk-1", "text": TEXT, "score": 0.8,
    "source_url": "https://kubernetes.io/docs/concepts/services-networking/service/",
    "source_path": "content/en/docs/concepts/services-networking/service.md",
    "section": "Service > Defining a Service",
}
OTHER = {**CONTEXT, "chunk_id": "chunk-2", "text": "Headless Services publish DNS records for individual Pods.",
         "source_path": "dns.md"}


def plan(count=1):
    return {"requirements": [{"question": "Какие порты использует Service?", "query": "Service port targetPort"},
                             {"question": "Какие DNS записи создаются?", "query": "headless Service DNS"}][:count]}


def step(quote=TEXT, requirement_id=1, source_id=1, next_query=None, next_id=None):
    return {"evidence": [{"requirement_id": requirement_id, "source_id": source_id, "quote": quote}],
            "next_requirement_id": next_id, "next_query": next_query}


def draft(source_ids=None, answer="Service: 80 → 9376 [1]."):
    return {"answer": answer, "source_ids": [1] if source_ids is None else source_ids}


def review(requirement_id=1, status="supported", source_ids=None, next_query=None):
    return {"answer": "Fixture conclusion" if status == "supported" else None,
            "requirement_id": requirement_id, "status": status,
            "source_ids": [1] if source_ids is None else source_ids,
            "reason": "Fixture judgment", "missing_information": None if status == "supported" else "Missing trigger",
            "next_query": next_query}


class FakeRetriever:
    collection = "test-collection"
    embedding_version = "test-version"

    def __init__(self, contexts=None):
        self.contexts = [CONTEXT] if contexts is None else contexts
        self.queries = []

    def retrieve(self, question, top_k):
        self.queries.append((question, top_k))
        return self.contexts[:top_k]


class FakeGenerator:
    config = GenerationConfig(verify_evidence=True)

    def __init__(self, responses=None, reviews=None):
        self.reviews = iter(reviews) if reviews is not None else None
        self.responses = iter([plan(), step(), draft()] if responses is None else responses)
        self.calls = []

    @staticmethod
    def count_tokens(messages):
        return sum(len(message["content"].split()) + 4 for message in messages)

    def generate(self, messages, schema=DraftAnswer):
        self.calls.append((schema, messages))
        if schema is EvidenceReview:
            if self.reviews is not None:
                response = next(self.reviews)
            else:
                payload = json.loads(messages[1]["content"])
                response = review(payload["requirement_id"], source_ids=[s["source_id"] for s in payload["sources"]])
        else:
            response = next(self.responses)
        return (response if isinstance(response, str) else json.dumps(response)), {"model": "glm4:9b"}


class RagTests(unittest.TestCase):
    def test_four_separate_phases_and_exact_citation(self):
        generator, retriever = FakeGenerator(), FakeRetriever()
        result = RagPipeline(retriever, generator).ask(" Какой порт? ", 5)
        self.assertEqual([schema for schema, _ in generator.calls], [SearchPlan, SearchStep, EvidenceReview, DraftAnswer])
        self.assertEqual(retriever.queries, [("Какой порт?", 5)])
        self.assertFalse(result.insufficient_context)
        self.assertEqual(result.citations[0].snippet, TEXT)
        self.assertIsNone(result.citations[0].page)
        self.assertEqual(result.metadata["stop_reason"], "requirements_covered")

    def test_plan_cannot_omit_an_explicit_user_question(self):
        questions = ["Какие порты использует Service?", "Какие DNS записи создаются?"]
        correct = plan(2)
        for item, question in zip(correct["requirements"], questions):
            item["question"] = question
        extraction = step()
        extraction["evidence"].append({"requirement_id": 2, "source_id": 2, "quote": OTHER["text"]})
        generator = FakeGenerator([plan(), correct, extraction, draft([1, 2], "Ответ [1] [2]")])
        result = RagPipeline(FakeRetriever([CONTEXT, OTHER]), generator).ask(" ".join(questions))
        self.assertEqual([r["question"] for r in result.metadata["requirements"]], questions)
        self.assertEqual(result.metadata["usage"]["planning"]["attempts"], 2)

    def test_coordinated_user_questions_are_preserved(self):
        generator = FakeGenerator([{"requirements": [
            {"question": "На какой порт обращается клиент", "query": "Service port"},
            {"question": "на какой порт Pod попадёт трафик?", "query": "Service targetPort"}]}])
        engine = RagPipeline(FakeRetriever(), generator)
        result, _ = engine._plan("На какой порт обращается клиент и на какой порт Pod попадёт трафик?")
        self.assertEqual(len(result.requirements), 2)

    def test_code_preserves_user_questions_if_planner_translates_them(self):
        generator = FakeGenerator([{"requirements": [
            {"question": "Which port does the client use?", "query": "Service port"},
            {"question": "Which port receives traffic?", "query": "Service targetPort"}]}])
        engine = RagPipeline(FakeRetriever(), generator)
        result, usage = engine._plan("Какой порт использует клиент? На какой порт Pod попадёт трафик?")
        self.assertEqual([r.question for r in result.requirements],
                         ["Какой порт использует клиент?", "На какой порт Pod попадёт трафик?"])
        self.assertEqual(usage["attempts"], 1)

    def test_one_quote_can_support_two_requirements_without_duplicate_citations(self):
        requirements = plan(2)
        requirements["requirements"][1]["question"] = "Как перенаправляется трафик Service?"
        extraction = step()
        extraction["evidence"].append({"requirement_id": 2, "source_id": 1, "quote": TEXT})
        generator = FakeGenerator([requirements, extraction, draft()])
        result = RagPipeline(FakeRetriever(), generator).ask("Вопрос")
        self.assertFalse(result.insufficient_context)
        self.assertEqual(len(result.citations), 1)

    def test_dependent_query_combines_evidence_from_two_documents(self):
        class ChainedRetriever(FakeRetriever):
            def retrieve(self, question, top_k):
                self.queries.append((question, top_k))
                return [CONTEXT] if len(self.queries) == 1 else [OTHER]

        generator = FakeGenerator([plan(2), step(next_query="headless Service DNS records", next_id=2),
                                   step(OTHER["text"], requirement_id=2),
                                   draft([1, 2], "Ответ [1] [2].")])
        retriever = ChainedRetriever()
        result = RagPipeline(retriever, generator).ask("Вопрос")
        self.assertEqual(retriever.queries[1][0], "headless Service DNS")
        self.assertEqual({c.chunk_id for c in result.citations}, {"chunk-1", "chunk-2"})
        self.assertEqual(len(result.metadata["hops"]), 2)
        self.assertFalse(result.insufficient_context)
        final_payload = json.loads(generator.calls[-1][1][1]["content"])
        self.assertEqual([s["text"] for s in final_payload["sources"]], [TEXT, OTHER["text"]])

    def test_null_does_not_end_search_with_missing_requirement(self):
        for query in (None, "null", " NONE ", "  "):
            with self.subTest(query=query):
                generator = FakeGenerator([plan(2), step(next_query=query), draft(answer="Частичный ответ [1].")])
                retriever = FakeRetriever()
                result = RagPipeline(retriever, generator).ask("Вопрос")
                self.assertEqual(retriever.queries[1][0], "headless Service DNS")
                self.assertEqual(result.metadata["missing_requirement_ids"], [2])
                self.assertTrue(result.insufficient_context)
                self.assertIn("Не подтверждено", result.answer)

    def test_all_requirements_can_be_supported_by_one_search(self):
        second = {"requirement_id": 2, "source_id": 2, "quote": OTHER["text"]}
        extraction = step()
        extraction["evidence"].append(second)
        generator = FakeGenerator([plan(2), extraction, draft([1, 2], "Ответ [1] [2]")])
        result = RagPipeline(FakeRetriever([CONTEXT, OTHER]), generator).ask("Вопрос")
        self.assertEqual(len(result.metadata["hops"]), 1)
        self.assertFalse(result.insufficient_context)

    def test_fabricated_quote_is_rejected_and_never_reaches_answer(self):
        generator = FakeGenerator([plan(), step("Invented unsupported storage deletion rule.")])
        result = RagPipeline(FakeRetriever(), generator, max_hops=1).ask("Вопрос")
        self.assertTrue(result.insufficient_context)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.metadata["evidence"], [])
        self.assertEqual(result.metadata["hops"][0]["rejected_evidence"][0]["reason"], "Quote is absent from source")
        self.assertEqual(len(generator.calls), 2)

    def test_quote_span_preserves_original_whitespace_and_unicode(self):
        source = {**CONTEXT, "text": "До цитаты.\n\nПривет\n  русский мир.\n\nПосле.", "source_id": 1}
        ledger = EvidenceLedger(SearchPlan.model_validate(plan()).requirements)
        quote = EvidenceQuote(requirement_id=1, source_id=1, quote="Привет русский мир.")
        accepted, rejected = ledger.add([quote], [source])
        self.assertEqual(rejected, [])
        entry = accepted[0]
        self.assertEqual(entry["quote"], "Привет\n  русский мир.")
        self.assertEqual(source["text"][entry["span_start"]:entry["span_end"]], entry["quote"])

    def test_quote_keeps_enclosing_policy_condition(self):
        text = "Intro.\n\n`Delete`\n: PVCs are removed when the StatefulSet is deleted.\n\n`Retain`\n: PVCs are kept."
        source = {**CONTEXT, "text": text, "source_id": 1}
        ledger = EvidenceLedger(SearchPlan.model_validate(plan()).requirements)
        quote = EvidenceQuote(requirement_id=1, source_id=1,
                              quote="PVCs are removed when the StatefulSet is deleted.")
        accepted, _ = ledger.add([quote], [source])
        self.assertTrue(accepted[0]["quote"].startswith("`Delete`\n:"))

    def test_related_quotes_keep_default_in_one_contiguous_source(self):
        text = "This field controls PVC retention.\n\nDelete removes PVCs.\n\nThe default is Retain."
        source = {**CONTEXT, "text": text, "source_id": 1}
        ledger = EvidenceLedger(SearchPlan.model_validate(plan()).requirements)
        quotes = [EvidenceQuote(requirement_id=1, source_id=1, quote=part)
                  for part in ("This field controls PVC retention.", "The default is Retain.")]
        accepted, rejected = ledger.add(quotes, [source])
        self.assertEqual(rejected, [])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["quote"], text)
        ledger.review(1, EvidenceReview.model_validate(review()), [{**accepted[0], "source_id": 1}])
        self.assertIn("The default is Retain.", ledger.sources()[0]["text"])

    def test_expanding_reviewed_context_requires_a_new_review(self):
        source = {**CONTEXT, "text": TEXT + "\n\nAn additional documented condition.", "source_id": 1}
        ledger = EvidenceLedger(SearchPlan.model_validate(plan()).requirements)
        ledger.add([EvidenceQuote(requirement_id=1, source_id=1, quote=TEXT),
                    EvidenceQuote(requirement_id=1, source_id=2, quote=OTHER["text"])],
                   [source, {**OTHER, "source_id": 2}])
        ledger.review(1, EvidenceReview.model_validate(review(source_ids=[1, 2])),
                      [{**e, "source_id": i} for i, e in enumerate(ledger.entries, 1)])
        ledger.add([EvidenceQuote(requirement_id=1, source_id=1, quote="An additional documented condition.")], [source])
        self.assertEqual(ledger.missing, [1])
        self.assertEqual(ledger.sources(), [])

    def test_wrong_sources_requirements_and_duplicate_quotes(self):
        ledger = EvidenceLedger(SearchPlan.model_validate(plan()).requirements)
        quotes = [EvidenceQuote(requirement_id=1, source_id=99, quote=TEXT),
                  EvidenceQuote(requirement_id=99, source_id=1, quote=TEXT)]
        accepted, rejected = ledger.add(quotes, [{**CONTEXT, "source_id": 1}])
        self.assertEqual(len(rejected), 2)
        self.assertEqual(accepted, [])
        quote = EvidenceQuote(requirement_id=1, source_id=1, quote=TEXT)
        ledger.add([quote, quote], [{**CONTEXT, "source_id": 1}])
        self.assertEqual(len(ledger.entries), 1)

    def test_hop_limit_returns_partial_evidence(self):
        generator = FakeGenerator([plan(2), step(), draft()])
        result = RagPipeline(FakeRetriever(), generator, max_hops=1).ask("Вопрос")
        self.assertEqual(len(result.metadata["hops"]), 1)
        self.assertEqual(result.metadata["stop_reason"], "hop_limit")
        self.assertTrue(result.insufficient_context)
        self.assertEqual(result.metadata["missing_requirement_ids"], [2])
        self.assertEqual(len(generator.calls), 3)
        self.assertIn(TEXT, result.answer)

    def test_final_answer_omitting_required_citation_is_incomplete(self):
        extraction = step()
        extraction["evidence"].append({"requirement_id": 2, "source_id": 2, "quote": OTHER["text"]})
        result = RagPipeline(FakeRetriever([CONTEXT, OTHER]), FakeGenerator([plan(2), extraction, draft()])).ask("Вопрос")
        self.assertTrue(result.insufficient_context)
        self.assertEqual(result.metadata["missing_requirement_ids"], [2])

    def test_final_sources_are_checked_and_invalid_answer_retried(self):
        generator = FakeGenerator([plan(), step(), draft([99], "Ответ [99]"), draft()])
        result = RagPipeline(FakeRetriever(), generator).ask("Вопрос")
        self.assertFalse(result.insufficient_context)
        self.assertEqual(result.metadata["usage"]["answer"]["attempts"], 2)

    def test_invalid_structured_plan_fails_after_two_attempts(self):
        generator = FakeGenerator(["{}", "{}"])
        with self.assertRaises(RuntimeError):
            RagPipeline(FakeRetriever(), generator).ask("Вопрос")

    def test_empty_retrieval_does_not_call_llm(self):
        generator = FakeGenerator([])
        result = RagPipeline(FakeRetriever([]), generator).ask("Вопрос")
        self.assertTrue(result.insufficient_context)
        self.assertEqual(generator.calls, [])

    def test_uncited_model_answer_becomes_abstention(self):
        result = RagPipeline(FakeRetriever(), FakeGenerator([plan(), step(), draft([], "Invented")])).ask("Вопрос")
        self.assertTrue(result.insufficient_context)
        self.assertNotIn("Invented", result.answer)

    def test_source_footer_and_reference_validation(self):
        result = RagPipeline(FakeRetriever(), FakeGenerator([plan(), step(), draft(answer="Ответ")])).ask("Вопрос")
        self.assertIn("Источники: [1]", result.answer)
        with self.assertRaises(ValueError):
            resolve_citations(DraftAnswer.model_validate(draft([1], "Ответ [2]")), [{**CONTEXT, "source_id": 1}])

    def test_prompt_respects_budget_without_cutting_sources(self):
        big = PromptBuilder(100000, FakeGenerator.count_tokens)
        first, _ = big.build("Вопрос", [CONTEXT])
        builder = PromptBuilder(big.size(first), FakeGenerator.count_tokens)
        messages, selected = builder.build("Вопрос", [CONTEXT, OTHER])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["text"], TEXT)
        self.assertLessEqual(builder.size(messages), builder.input_budget)

    def test_optional_answer_review_uses_only_quotes(self):
        generator = FakeGenerator([plan(), step(), draft(), draft(answer="Проверено [1]")])
        generator.config = replace(generator.config, verify_answers=True)
        result = RagPipeline(FakeRetriever(), generator).ask("Вопрос")
        self.assertEqual(result.answer, "Проверено [1]")
        self.assertIn("verification", result.metadata["usage"]["answer"])

    def test_api_validation_and_response(self):
        @contextmanager
        def factory():
            yield RagPipeline(FakeRetriever(), FakeGenerator())

        with TestClient(create_app(factory)) as client:
            self.assertEqual(client.get("/health").json()["llm_model"], "glm4:9b")
            self.assertEqual(client.post("/ask", json={"question": "   "}).status_code, 422)
            self.assertEqual(client.post("/ask", json={"question": "Вопрос", "top_k": 0}).status_code, 422)
            response = client.post("/ask", json={"question": "Вопрос"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["citations"][0]["chunk_id"], "chunk-1")

    def test_api_reports_backend_and_model_errors(self):
        class UnavailableGenerator(FakeGenerator):
            def generate(self, messages, schema=DraftAnswer):
                raise httpx.ConnectError("Ollama offline")

        for generator, status in ((UnavailableGenerator(), 503), (FakeGenerator(["{}", "{}"]), 502)):
            @contextmanager
            def factory():
                yield RagPipeline(FakeRetriever(), generator)

            with TestClient(create_app(factory)) as client:
                self.assertEqual(client.post("/ask", json={"question": "Вопрос"}).status_code, status)

    def test_ollama_uses_requested_structured_schema(self):
        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "glm4:9b")
            self.assertIn("requirements", payload["format"]["properties"])
            self.assertFalse(payload["truncate"])
            self.assertFalse(payload["stream"])
            return httpx.Response(200, json={"done": True, "message": {"content": json.dumps(plan())},
                                             "prompt_eval_count": 12})

        with httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler)) as client, \
                patch.object(ChatTokenCounter, "count", return_value=12):
            content, _ = OllamaGenerator(GenerationConfig(), client).generate([], schema=SearchPlan)
            self.assertEqual(len(SearchPlan.model_validate_json(content).requirements), 1)

    def test_truncated_output_is_rejected(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"done_reason": "length"}))
        with httpx.Client(base_url="http://ollama", transport=transport) as client, \
                patch.object(ChatTokenCounter, "count", return_value=12):
            with self.assertRaises(RuntimeError):
                OllamaGenerator(GenerationConfig(), client).generate([])

    def test_retrieval_score_threshold_filters_unrelated_contexts(self):
        retriever = object.__new__(DocumentRetriever)
        retriever.min_score = 0.5
        retriever.encoder = SimpleNamespace(encode_query=lambda question: [1.0])
        retriever.search = SimpleNamespace(query=lambda vector, limit: [
            SimpleNamespace(score=0.7, payload={"chunk_id": "chunk-1"}),
            SimpleNamespace(score=0.38, payload={"chunk_id": "irrelevant"}),
        ])
        retriever.chunks = {"chunk-1": {**CONTEXT, "heading_path": ["Service"]}}
        self.assertEqual(len(retriever.retrieve("Вопрос", 10)), 1)

    def test_default_mode_keeps_previous_pipeline_without_critic(self):
        generator = FakeGenerator()
        generator.config = GenerationConfig()
        result = RagPipeline(FakeRetriever(), generator).ask("Вопрос")
        self.assertFalse(result.insufficient_context)
        self.assertEqual([schema for schema, _ in generator.calls], [SearchPlan, SearchStep, DraftAnswer])
        self.assertEqual(result.metadata["evidence_reviews"], [])
        self.assertFalse(result.metadata["generation"]["verify_evidence"])

    def test_real_quote_is_not_coverage_until_reviewed(self):
        ledger = EvidenceLedger(SearchPlan.model_validate(plan()).requirements)
        ledger.add([EvidenceQuote(requirement_id=1, source_id=1, quote=TEXT)], [{**CONTEXT, "source_id": 1}])
        self.assertEqual(ledger.missing, [1])
        self.assertEqual(ledger.sources(), [])

    def test_related_rollout_quote_fails_review_and_drives_next_search(self):
        generic = "Deployments progressively update the Pods, limiting old and new replica counts."
        context = {**CONTEXT, "text": generic}
        generator = FakeGenerator([plan(), step(generic)], reviews=[review(status="unsupported",
            next_query="Deployment rollout triggered only spec.template scaling does not trigger")])
        result = RagPipeline(FakeRetriever([context]), generator, max_hops=2).ask("Как запустить rollout?")
        self.assertTrue(result.insufficient_context)
        self.assertEqual(result.citations, [])
        self.assertEqual(result.metadata["evidence"], [])
        self.assertEqual(result.metadata["hops"][0]["next_query_origin"], "critic")
        self.assertIn("spec.template", result.metadata["hops"][1]["query"])
        self.assertNotIn(DraftAnswer, [schema for schema, _ in generator.calls])
        payload = json.loads(generator.calls[2][1][1]["content"])
        self.assertEqual(payload["sources"][0]["text"], generic)
        self.assertNotIn("previous_answer", payload)
        self.assertEqual(payload["question"], plan()["requirements"][0]["question"])
        self.assertEqual(payload["original_question"], "Как запустить rollout?")

    def test_review_rejection_can_be_resolved_by_second_hop(self):
        class ChainedRetriever(FakeRetriever):
            def retrieve(self, question, top_k):
                self.queries.append((question, top_k))
                return [CONTEXT] if len(self.queries) == 1 else [OTHER]

        generator = FakeGenerator([plan(), step(), step(OTHER["text"]), draft()],
            reviews=[review(status="partial", next_query="missing documented rule"), review(source_ids=[2])])
        result = RagPipeline(ChainedRetriever(), generator, max_hops=2).ask("Вопрос")
        self.assertFalse(result.insufficient_context)
        self.assertEqual(len(result.metadata["hops"]), 2)
        self.assertEqual([c.chunk_id for c in result.citations], ["chunk-2"])
        self.assertEqual(len(result.metadata["hops"][1]["reviews"][0]["reviewed_evidence_ids"]), 2)

    def test_partial_review_does_not_close_requirement(self):
        generator = FakeGenerator([plan(), step()], reviews=[review(status="partial")])
        result = RagPipeline(FakeRetriever(), generator, max_hops=1).ask("Вопрос")
        self.assertEqual(result.metadata["missing_requirement_ids"], [1])
        self.assertEqual(result.citations, [])

    def test_review_cannot_approve_wrong_requirement_or_unknown_source(self):
        for invalid in (review(requirement_id=2), review(source_ids=[99]), review(source_ids=[]),
                        {**review(), "answer": None}):
            with self.subTest(invalid=invalid):
                generator = FakeGenerator([plan(), step(), draft()], reviews=[invalid, review()])
                result = RagPipeline(FakeRetriever(), generator).ask("Вопрос")
                self.assertFalse(result.insufficient_context)
                self.assertEqual(result.metadata["hops"][0]["reviews"][0]["usage"]["attempts"], 2)

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            DocumentRetriever(None, "missing-bundle", min_score=1.1)
        with self.assertRaises(ValueError):
            RagPipeline(FakeRetriever(), FakeGenerator(), max_hops=0)


if __name__ == "__main__":
    unittest.main()
