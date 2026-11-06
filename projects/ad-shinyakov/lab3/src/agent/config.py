from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 16
    timeout_seconds: float = 600
    context_limit_tokens: int = 8192
    model_context_window: int = 32768
    max_output_tokens: int = 2048
    model: str = 'glm4-agent:9b'
    ollama_url: str = 'http://localhost:11434'

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError('model must be a non-empty string')
        if not isinstance(self.ollama_url, str) or not self.ollama_url.startswith(('http://', 'https://')):
            raise ValueError('ollama_url must be an HTTP(S) URL')
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError('max_steps must be a positive integer')
        if isinstance(self.timeout_seconds, bool) or not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be positive and finite')
        for name in ('context_limit_tokens', 'model_context_window', 'max_output_tokens'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.max_output_tokens >= self.model_context_window:
            raise ValueError('max_output_tokens must be smaller than model_context_window')

    @property
    def input_budget(self):
        return min(self.context_limit_tokens, self.model_context_window - self.max_output_tokens)
