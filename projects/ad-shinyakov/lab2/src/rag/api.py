from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
import httpx
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from .schemas import Question, RagAnswer


def create_app(engine_factory):
    @asynccontextmanager
    async def lifespan(app):
        with engine_factory() as engine:
            app.state.engine = engine
            yield

    app = FastAPI(title="Kubernetes RAG", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        engine = request.app.state.engine
        return {"status": "ok", "collection": engine.retriever.collection,
                "llm_model": engine.generator.config.model}

    @app.post("/ask", response_model=RagAnswer)
    def ask(question: Question, request: Request):
        try:
            return request.app.state.engine.ask(question.question, question.top_k)
        except httpx.TimeoutException as error:
            raise HTTPException(504, "LLM request timed out") from error
        except (httpx.HTTPError, ResponseHandlingException, UnexpectedResponse) as error:
            raise HTTPException(503, f"LLM or vector store unavailable: {error}") from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(502, str(error)) from error

    return app
