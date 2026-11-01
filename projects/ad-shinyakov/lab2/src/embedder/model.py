from dataclasses import replace
from pathlib import Path

import numpy as np


class SentenceEncoder:
    def __init__(self, config, device=None):
        self.config = config
        self.device = device
        self.model = None

    def resolve(self):
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(self.config.model, revision=self.config.revision, allow_patterns=["config.json"])
        self.config = replace(self.config, revision=Path(snapshot).name)
        return self.config

    def load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.config.model, revision=self.config.revision, device=self.device, trust_remote_code=False)
            self.model.max_seq_length = self.config.max_length

    def encode_documents(self, texts, batch_size):
        self.load()
        return self._encode([self.config.document_prefix + text for text in texts], batch_size)

    def encode_query(self, text):
        return self.encode_queries([text])[0]

    def encode_queries(self, texts, batch_size=8):
        self.load()
        return self._encode([self.config.query_prefix + text for text in texts], batch_size)

    def _encode(self, texts, batch_size):
        for index, text in enumerate(texts):
            length = len(self.model.tokenizer.encode(text, add_special_tokens=True))
            if length > self.config.max_length:
                raise ValueError(f"Input {index} has {length} tokens, exceeding --max-length={self.config.max_length}; refusing silent truncation")
        dimension = self.model.get_sentence_embedding_dimension()
        if self.config.dimensions is not None and self.config.dimensions > dimension:
            raise ValueError("Requested dimensions exceed model output")
        vectors = self.model.encode(
            texts, batch_size=batch_size, prompt="", convert_to_numpy=True,
            normalize_embeddings=True, truncate_dim=self.config.dimensions,
            show_progress_bar=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Model produced non-finite or zero vectors")
        return vectors
