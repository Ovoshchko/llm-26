import argparse
import os
from pathlib import Path
import sys

import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from .llm import GenerationConfig
from .service import open_pipeline


def print_answer(answer, as_json=False):
    if as_json:
        print(answer.model_dump_json(indent=2))
        return
    print(answer.answer)
    for citation in answer.citations:
        print(f"\n[{citation.id}] {citation.section}\n{citation.source_url}\n{citation.snippet}")


def main():
    parser = argparse.ArgumentParser(description="Kubernetes RAG with citations and Ollama")
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--collection")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL", "glm4:9b"))
    parser.add_argument("--context-window", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--verify-answer", action="store_true")
    parser.add_argument("--verify-evidence", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-hops", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--device")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--question")
    mode.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        generation = GenerationConfig(args.llm_model, args.ollama_url, args.context_window, args.max_tokens,
                                      args.timeout, verify_answers=args.verify_answer,
                                      verify_evidence=args.verify_evidence)

        def factory():
            return open_pipeline(
                args.embeddings, generation, args.collection, args.qdrant_url,
                os.environ.get("QDRANT_API_KEY"), args.device, args.top_k, args.ef_search, args.min_score, args.max_hops,
            )

        if args.serve:
            import uvicorn
            from .api import create_app

            uvicorn.run(create_app(factory), host=args.host, port=args.port, workers=1)
        else:
            with factory() as engine:
                if args.question is not None:
                    print_answer(engine.ask(args.question), args.json)
                else:
                    interactive(engine, args.json)
        return 0
    except (OSError, ValueError, RuntimeError, httpx.HTTPError, ResponseHandlingException, UnexpectedResponse) as error:
        print(str(error), file=sys.stderr)
        return 1


def interactive(engine, as_json):
    print("Вопросы по Kubernetes. Для выхода: /exit. Каждый вопрос ищется независимо.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question == "/exit":
            return
        if question:
            print_answer(engine.ask(question), as_json)
