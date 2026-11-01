from dataclasses import dataclass

import httpx

from .schemas import DraftAnswer
from .tokenizer import ChatTokenCounter


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "glm4:9b"
    url: str = "http://localhost:11434"
    context_window: int = 32768
    max_tokens: int = 1024
    timeout: float = 180
    verify_answers: bool = False
    verify_evidence: bool = False

    def __post_init__(self):
        if self.max_tokens <= 0 or self.context_window <= self.max_tokens + 1024 or self.timeout <= 0:
            raise ValueError("Invalid LLM context, output limit or timeout")


class OllamaGenerator:
    def __init__(self, config, client=None):
        self.config = config
        self.client = client or httpx.Client(base_url=config.url.rstrip("/"), timeout=config.timeout)
        self.counter = ChatTokenCounter()

    def count_tokens(self, messages):
        return self.counter.count(messages)

    def generate(self, messages, schema=DraftAnswer):
        input_tokens = self.count_tokens(messages)
        if input_tokens + self.config.max_tokens > self.config.context_window:
            raise ValueError("Prompt plus max_tokens exceeds the LLM context window")
        response = self.client.post("/api/chat", json={
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "truncate": False,
            "shift": False,
            "format": schema.model_json_schema(),
            "options": {"temperature": 0, "num_ctx": self.config.context_window,
                        "num_predict": self.config.max_tokens, "seed": 42},
            "keep_alive": "10m",
        })
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as error:
            raise RuntimeError("Ollama returned invalid JSON") from error
        if data.get("done_reason") == "length":
            raise RuntimeError("LLM output exceeded max_tokens; increase --max-tokens")
        if not data.get("done") or not data.get("message", {}).get("content"):
            raise RuntimeError("Ollama returned an incomplete or empty answer")
        if data.get("prompt_eval_count") != input_tokens:
            raise RuntimeError("Ollama prompt token count differs from the preflight count")
        usage = {k: data[k] for k in ("model", "prompt_eval_count", "eval_count", "total_duration") if k in data}
        return data["message"]["content"], usage

    def close(self):
        self.client.close()
