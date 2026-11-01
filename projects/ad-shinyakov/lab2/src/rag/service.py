from contextlib import closing, contextmanager

from qdrant_client import QdrantClient

from .llm import OllamaGenerator
from .pipeline import RagPipeline
from .retriever import DocumentRetriever


@contextmanager
def open_pipeline(embeddings, generation, collection=None, qdrant_url="http://localhost:6333",
                  qdrant_api_key=None, device=None, top_k=10, ef_search=128, min_score=0.5, max_hops=3):
    with closing(QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)) as client:
        with closing(OllamaGenerator(generation)) as generator:
            retriever = DocumentRetriever(client, embeddings, collection, ef_search, min_score, device)
            yield RagPipeline(retriever, generator, top_k, max_hops)
