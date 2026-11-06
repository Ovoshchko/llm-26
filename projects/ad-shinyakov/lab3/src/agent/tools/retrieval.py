import sys
from threading import Lock
from typing import Annotated

from pydantic import Field

from .config import LAB2_ROOT, ToolConfig


class DocumentationSearch:
    def __init__(self, config: ToolConfig):
        self.config = config
        self.client = None
        self.retriever = None
        self.lock = Lock()

    def _load(self):
        if self.retriever is not None:
            return
        source = str(LAB2_ROOT / 'src')
        if source not in sys.path:
            sys.path.insert(0, source)
        from qdrant_client import QdrantClient
        from rag.retriever import DocumentRetriever

        self.client = QdrantClient(url=self.config.qdrant_url, timeout=self.config.request_timeout)
        try:
            self.retriever = DocumentRetriever(self.client, self.config.embeddings)
        except Exception:
            self.client.close()
            self.client = None
            raise

    def search(self, query: Annotated[str, Field(min_length=1, max_length=1000)],
               top_k: Annotated[int, Field(ge=1, le=5)] = 3) -> dict:
        with self.lock:
            self._load()
            passages = self.retriever.retrieve(query, top_k)
        return {'query': query, 'passages': [self._passage(passage) for passage in passages]}

    @staticmethod
    def _passage(passage: dict) -> dict:
        return {**passage, 'text': passage['text'][:2000], 'truncated': len(passage['text']) > 2000}

    def close(self):
        if self.client is not None:
            self.client.close()
