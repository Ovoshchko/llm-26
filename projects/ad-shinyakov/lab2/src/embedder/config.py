from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    revision: str = "main"
    max_length: int = 2048
    dimensions: int = None
    document_prefix: str = ""
    query_prefix: str = "Instruct: Retrieve Kubernetes documentation passages relevant to the question.\nQuery: "

    def __post_init__(self):
        if self.max_length <= 0 or (self.dimensions is not None and self.dimensions <= 0):
            raise ValueError("max_length and dimensions must be positive")
