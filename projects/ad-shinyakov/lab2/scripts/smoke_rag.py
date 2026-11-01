import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx

from artifacts import write_json
from embedder import EmbeddingBundle
from rag.schemas import RagAnswer


CASES = [
    ("single_hop", "В Service заданы port: 80 и targetPort: 9376. На какой порт обращается клиент и на какой порт Pod попадёт трафик?", 3),
    ("multi_hop", "Есть StatefulSet db с serviceName=db-headless в namespace prod, обычной нумерацией с нуля и доменом cluster.local. Headless Service имеет именованный TCP-порт sql:5432. Какое полное DNS-имя будет у первого Pod и какое имя SRV-запроса использовать для sql?", 5),
    ("out_of_scope", "Как приготовить шарлотку с яблоками?", 3),
]


def verify(kind, answer, bundle):
    chunks = {c["id"]: c for c in bundle.chunks}
    if answer.metadata["embedding_version"] != bundle.version:
        raise ValueError("API and smoke test use different embeddings")
    for citation in answer.citations:
        chunk = chunks[citation.chunk_id]
        if " ".join(citation.snippet.split()) not in " ".join(chunk["text"].split()):
            raise ValueError("Citation snippet is absent from source text")
        if citation.source_url != chunk["source_url"] or citation.source_path != chunk["source_path"]:
            raise ValueError("Citation source metadata differs from bundle")
        if answer.metadata["context_chunk_ids"][citation.id - 1] != citation.chunk_id:
            raise ValueError("Citation ID differs from the supplied source")
    if kind == "out_of_scope":
        return answer.insufficient_context and not answer.citations
    if answer.insufficient_context or not answer.citations:
        return False
    if kind == "single_hop":
        return "80" in answer.answer and "9376" in answer.answer
    return ("db-0.db-headless.prod.svc.cluster.local" in answer.answer
            and "_sql._tcp.db-headless.prod.svc.cluster.local" in answer.answer
            and len({c.source_path for c in answer.citations}) >= 2)


def write_report(report, output):
    lines = [
        "# Сквозная проверка пункта 6", "",
        f"LLM из первой лабораторной: `{report['health']['llm_model']}` через Ollama.",
        f"Коллекция: `{report['health']['collection']}`. Использован настоящий Qwen-эмбеддер и Qdrant.",
        "Проверка по HTTP API: вопрос → вектор → поиск → контекст → LLM → ответ с источниками.", "",
        "| Сценарий | top-k | HTTP | Проверка | Время, с |",
        "|---|---:|---:|---|---:|",
    ]
    for case in report["cases"]:
        lines.append(f"| {case['type']} | {case['top_k']} | {case['status_code']} | {case['passed']} | {case['seconds']:.1f} |")
    lines.extend(["", "Все сниппеты и URL сверены с проверенным embedding bundle.",
                  "Для multi-hop дополнительно проверены оба DNS-имени и цитирование минимум двух документов.",
                  "Это три smoke-сценария, не полная оценка качества QA на 20 вопросах.", ""])
    for case in report["cases"]:
        lines.extend([f"## {case['type']}", "", case["question"], ""])
        if case["status_code"] == 200:
            result = case["response"]
            lines.extend([result["answer"], ""])
            for citation in result["citations"]:
                lines.append(f"- [{citation['id']}] [{citation['section']}]({citation['source_url']})")
            lines.append("")
        else:
            lines.extend([str(case["response"]), ""])
    lines.extend(["Полные ответы, сниппеты, ID, время, usage и настройки: [results.json](results.json).", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test a running RAG API against real source chunks")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output or project / "evaluation/local_runs/rag" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_rag_smoke")
    output.mkdir(parents=True, exist_ok=True)
    bundle = EmbeddingBundle(args.embeddings)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "cases": []}
    with httpx.Client(base_url=args.url, timeout=400) as client:
        health = client.get("/health")
        health.raise_for_status()
        report["health"] = health.json()
        for kind, question, top_k in CASES:
            print(f"Running {kind}", flush=True)
            started = perf_counter()
            response = client.post("/ask", json={"question": question, "top_k": top_k})
            case = {"type": kind, "question": question, "top_k": top_k,
                    "status_code": response.status_code, "seconds": perf_counter() - started,
                    "response": response.json(), "passed": False}
            if response.is_success:
                case["passed"] = verify(kind, RagAnswer.model_validate(case["response"]), bundle)
            report["cases"].append(case)
            write_json(output / "results.json", report)
            write_report(report, output)
            print(f"{kind}: HTTP {response.status_code}, passed={case['passed']}, {case['seconds']:.1f}s", flush=True)
    print(f"Saved to {output}")
    return 0 if all(c["passed"] for c in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
