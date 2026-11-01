import logging
import re
from itertools import zip_longest
from threading import Lock
from time import perf_counter

from pydantic import ValidationError

from .evidence import EvidenceLedger
from .prompt import ANSWER, EXTRACT, PLAN, REVIEW, PromptBuilder
from .schemas import Citation, DraftAnswer, EvidenceReview, Question, RagAnswer, SearchPlan, SearchStep


INSUFFICIENT = "В найденной документации недостаточно информации для подтверждённого ответа. Уточните вопрос или расширьте корпус."


def covered_requirements(sources):
    return {r for s in sources for r in s.get("requirement_ids", [s["requirement_id"]])}


def resolve_citations(draft, contexts):
    sources = {c["source_id"]: c for c in contexts}
    inline = {int(i) for i in re.findall(r"\[(\d+)\]", draft.answer)}
    ids = list(dict.fromkeys(draft.source_ids))
    if not inline.issubset(ids):
        raise ValueError("Inline references must be included in source_ids")
    citations = []
    for source_id in ids:
        source = sources.get(source_id)
        if source is None:
            raise ValueError(f"Source [{source_id}] is outside the supplied evidence")
        citations.append(Citation(
            id=source_id, chunk_id=source["chunk_id"], source_url=source["source_url"],
            source_path=source["source_path"], section=source["section"],
            snippet=source["text"], score=source["score"],
        ))
    return citations


class RagPipeline:
    def __init__(self, retriever, generator, top_k=10, max_hops=3):
        if not 1 <= top_k <= 20 or not 1 <= max_hops <= 5:
            raise ValueError("top_k must be 1-20 and max_hops must be 1-5")
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.max_hops = max_hops
        self.input_budget = generator.config.context_window - generator.config.max_tokens
        self.prompt = PromptBuilder(self.input_budget, generator.count_tokens)
        self.lock = Lock()

    def ask(self, question, top_k=None):
        request = Question(question=question, top_k=top_k)
        with self.lock:
            return self._answer(request)

    def _answer(self, request):
        started = perf_counter()
        top_k = request.top_k or self.top_k
        hits = self.retriever.retrieve(request.question, top_k)
        retrieval_seconds = perf_counter() - started
        ledger, trace, usage, selected = EvidenceLedger([]), [], {}, []
        answer, citations, missing = INSUFFICIENT, [], []
        stop_reason, context_count = "no_context", 0
        if hits:
            plan, usage["planning"] = self._plan(request.question)
            ledger = EvidenceLedger(plan.requirements, verify=self.generator.config.verify_evidence)
            trace, context_count, search_seconds, stop_reason = self._collect(request, plan, ledger, hits)
            retrieval_seconds += search_seconds
            if ledger.approved:
                answer, citations, selected, usage["answer"] = self._synthesize(request.question, plan, ledger)
            cited = {c.id for c in citations}
            covered = covered_requirements([s for s in selected if s["source_id"] in cited])
            missing = [i for i in range(1, len(plan.requirements) + 1) if i not in covered]
            if citations and missing:
                answer = "Найдены только частичные подтверждения:\n\n" + "\n\n".join(
                    f"[{c.id}] {c.snippet}" for c in citations)
                answer += "\n\nНе подтверждено источниками: " + "; ".join(
                    plan.requirements[i - 1].question for i in missing)
        requirements = [{"id": i, **r.model_dump()} for i, r in enumerate(ledger.requirements, 1)]
        return RagAnswer(
            question=request.question, answer=answer, citations=citations,
            insufficient_context=not citations or bool(missing),
            metadata={"architecture": ("verified-evidence-multihop-v2" if self.generator.config.verify_evidence
                                       else "planned-evidence-multihop-v1"), "llm_model": self.generator.config.model,
                      "generation": {"context_window": self.generator.config.context_window,
                                     "max_tokens": self.generator.config.max_tokens, "temperature": 0,
                                     "seed": 42, "verify_answers": self.generator.config.verify_answers,
                                     "verify_evidence": self.generator.config.verify_evidence},
                      "collection": self.retriever.collection,
                      "embedding_version": self.retriever.embedding_version,
                      "requirements": requirements, "missing_requirement_ids": missing,
                      "coverage_method": ("exact quotes plus separate same-model sufficiency review; not a correctness guarantee"
                          if self.generator.config.verify_evidence else "model-selected exact quotes; semantic entailment is not verified"),
                      "evidence": [{k: v for k, v in e.items() if k != "text"}
                                   for e in ledger.entries if e["evidence_id"] in ledger.approved],
                      "evidence_reviews": list(ledger.reviews.values()),
                      "max_hops": self.max_hops, "hops": trace, "stop_reason": stop_reason,
                      "retrieved_chunks": context_count, "context_chunks": len(selected),
                      "context_chunk_ids": [c["chunk_id"] for c in selected],
                      "retrieval_seconds": retrieval_seconds, "total_seconds": perf_counter() - started,
                      "usage": usage},
        )

    def _plan(self, question):
        questions = []
        for sentence in re.split(r"(?<=[.!?])\s+", question):
            if sentence.strip().endswith("?"):
                questions.extend(re.split(
                    r"\s+и\s+(?=(?:как|почему|что|где|когда|куда|зачем|сколько|(?:на\s+)?како\w*)\b)",
                    sentence.strip(), flags=re.IGNORECASE))
        required = questions if 2 <= len(questions) <= 4 else []
        messages, _ = self.prompt.build(question, [], PLAN, {"required_questions": required})

        def validate(plan):
            if required:
                if len(plan.requirements) != len(required):
                    raise ValueError("Return one requirement per required_questions item, in order; do not merge or add tasks")
                for requirement, original in zip(plan.requirements, required):
                    requirement.question = original

        return self._call(messages, SearchPlan, validate)

    def _collect(self, request, plan, ledger, first_hits):
        contexts, trace, seen = [], [], set()
        query, target, search_seconds, stop_reason = request.question, 1, 0.0, "hop_limit"
        requirements = [{"id": i, **r.model_dump()} for i, r in enumerate(plan.requirements, 1)]
        for hop in range(self.max_hops):
            if hop:
                started = perf_counter()
                hits = self.retriever.retrieve(query, request.top_k or self.top_k)
                search_seconds += perf_counter() - started
            else:
                hits = first_hits
            seen.add(query.strip().casefold())
            known = {c["chunk_id"] for c in contexts}
            fresh = [c for c in hits if c["chunk_id"] not in known]
            contexts = [c for pair in zip_longest(fresh, contexts) for c in pair if c is not None]
            record = {"query": query, "target_requirement_id": target,
                      "chunk_ids": [c["chunk_id"] for c in hits], "new_chunks": len(fresh)}
            trace.append(record)
            step = None
            if fresh:
                data = {"requirements": requirements, "target_requirement_id": target,
                        "missing_requirement_ids": ledger.missing,
                        "known_evidence": [{"requirement_id": e["requirement_id"], "quote": e["quote"]}
                                           for e in ledger.entries if e["evidence_id"] in ledger.approved],
                        "evidence_reviews": list(ledger.reviews.values())}
                messages, selected = self.prompt.build(request.question, contexts, EXTRACT, data)
                if not selected:
                    stop_reason = "context_budget"
                    break
                step, record["usage"] = self._call(messages, SearchStep)
                accepted, rejected = ledger.add(step.evidence, selected)
                record.update(evidence=[{k: v for k, v in e.items() if k != "text"} for e in accepted],
                              rejected_evidence=rejected, context_chunk_ids=[s["chunk_id"] for s in selected],
                              next_query=step.next_query)
                record["reviews"] = (self._review(request.question, plan, ledger, accepted)
                                     if self.generator.config.verify_evidence else [])
            record["missing_requirement_ids"] = ledger.missing
            if not ledger.missing:
                stop_reason = "requirements_covered"
                break
            correction = next(((i, ledger.reviews[i]["next_query"]) for i in ledger.missing
                               if i in ledger.reviews and ledger.reviews[i]["next_query"]
                               and ledger.reviews[i]["next_query"].casefold() not in seen), None)
            if correction:
                target, query = correction
                record.update(next_query=query, next_query_origin="critic")
                continue
            fallback = next(((i, plan.requirements[i - 1].query) for i in ledger.missing
                             if plan.requirements[i - 1].query.strip().casefold() not in seen), None)
            if fallback is not None:
                target, query = fallback
                record.update(next_query=query, next_query_origin="plan")
                continue
            if step and step.next_requirement_id in ledger.missing and step.next_query:
                if step.next_query.casefold() not in seen:
                    query, target = step.next_query, step.next_requirement_id
                    record.update(next_query=query, next_query_origin="model")
                    continue
            stop_reason = "no_progress"
            break
        return trace, len(contexts), search_seconds, stop_reason

    def _review(self, question, plan, ledger, accepted):
        reviews = []
        for requirement_id in sorted({e["requirement_id"] for e in accepted} & set(ledger.missing)):
            candidates = [{**e, "text": e["quote"]} for e in ledger.entries
                          if e["requirement_id"] == requirement_id]
            messages, sources = self.prompt.build(plan.requirements[requirement_id - 1].question, candidates, REVIEW, {
                "original_question": question,
                "requirement_id": requirement_id,
                "lookup_query": plan.requirements[requirement_id - 1].query,
                "requirement": plan.requirements[requirement_id - 1].question})
            if not sources:
                continue
            _, usage = self._call(messages, EvidenceReview,
                lambda r: ledger.review(requirement_id, r, sources))
            reviews.append({**ledger.reviews[requirement_id], "usage": usage,
                            "reviewed_evidence_ids": [s["evidence_id"] for s in sources]})
        return reviews

    def _synthesize(self, question, plan, ledger):
        data = {"requirements": [{"id": i, **r.model_dump()} for i, r in enumerate(plan.requirements, 1)],
                "missing_requirement_ids": ledger.missing}
        messages, selected = self.prompt.build(question, ledger.sources(), ANSWER, data)
        if not selected:
            return INSUFFICIENT, [], [], {}
        if ledger.missing or len(covered_requirements(selected)) < len(plan.requirements):
            draft = DraftAnswer(answer="Частичные подтверждения",
                source_ids=[s["source_id"] for s in selected])
            return draft.answer, resolve_citations(draft, selected), selected, {"mode": "extractive_partial"}
        draft, usage = self._call(messages, DraftAnswer, lambda d: resolve_citations(d, selected))
        if self.generator.config.verify_answers:
            data["previous_answer"] = draft.answer
            messages, selected = self.prompt.build(question, ledger.sources(), ANSWER +
                " Review the previous answer. Remove every claim not established by the supplied quotes.", data)
            draft, usage["verification"] = self._call(messages, DraftAnswer,
                                                     lambda d: resolve_citations(d, selected))
        citations = resolve_citations(draft, selected)
        if not citations:
            return INSUFFICIENT, [], selected, usage
        inline = {int(i) for i in re.findall(r"\[(\d+)\]", draft.answer)}
        missing = [c.id for c in citations if c.id not in inline]
        answer = draft.answer
        if missing:
            answer += "\n\nИсточники: " + " ".join(f"[{i}]" for i in missing)
        return answer, citations, selected, usage

    def _call(self, messages, schema, validate=None):
        calls = []
        for attempt in range(2):
            if self.prompt.size(messages) > self.input_budget:
                raise ValueError("Structured request exceeds the LLM input budget")
            content, usage = self.generator.generate(messages, schema=schema)
            calls.append(usage)
            try:
                result = schema.model_validate_json(content)
                if validate:
                    validate(result)
                return result, {"attempts": attempt + 1, "calls": calls}
            except (ValidationError, ValueError) as error:
                logging.getLogger(__name__).warning("LLM %s validation failed: %s", schema.__name__, error)
                if attempt:
                    raise RuntimeError(f"LLM failed {schema.__name__} validation after two attempts") from error
                messages = messages + [{"role": "user", "content": "Correct the JSON or source references: " + str(error)[:300]}]
        raise RuntimeError("No structured LLM response")
