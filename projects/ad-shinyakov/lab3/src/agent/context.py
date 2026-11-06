from collections.abc import Sequence
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .config import AgentConfig
from .memory import MemoryCompressor, memory_message
from .tokenizer import ChatTokenCounter, TokenCounter


class ContextManager:
    def __init__(self, config: AgentConfig, token_counter: TokenCounter | None = None,
                 system_prompt: str | None = None, tools=(), compressor: MemoryCompressor | None = None):
        self.config = config
        template = (Path(__file__).resolve().parents[2] / 'prompts/chat_template.jinja').read_text()
        self.counter = token_counter if token_counter is not None else ChatTokenCounter(
            chat_template=template if config.model == 'glm4-agent:9b' else None)
        self.system_prompt = system_prompt if system_prompt is not None else (
            Path(__file__).resolve().parents[2] / 'prompts/system.md').read_text()
        self.tools = list(tools)
        self.compressor = compressor if compressor is not None else MemoryCompressor(config, self.counter)

    def count(self, messages: list[BaseMessage]) -> int:
        return self.counter.count(messages, tools=self.tools)

    def prepare(self, messages: Sequence[BaseMessage], memory=None, config=None) -> dict:
        history = list(messages)
        offset = int(bool(history) and isinstance(history[0], SystemMessage))
        system = history[0] if offset else SystemMessage(self.system_prompt)
        if len(history) <= offset or not isinstance(history[offset], HumanMessage):
            return self._state_update([], 0, 0, 0, 'invalid_history')
        return self._prepare_memory(history, system, memory, config)

    def _prepare_memory(self, history, system, memory, config):
        offset = int(isinstance(history[0], SystemMessage))
        covered = memory['covered_messages'] if memory else offset
        if memory and memory.get('strategy') != self.compressor.strategy:
            covered, memory = offset, None
        selected = [system, *([memory_message(memory)] if memory else []), *history[covered:]]
        before = self.count(selected)
        if before <= self.config.input_budget:
            return self._state_update(selected, before, before, covered - offset)
        current = max(i for i, message in enumerate(history) if isinstance(message, HumanMessage))
        if current <= offset:
            return self._state_update([], before, before, covered - offset, 'context_limit')
        try:
            fits = lambda value: self.count([system, memory_message(value), *history[current:]]) <= self.config.input_budget
            updated = self.compressor.compress(history, current, fits=fits)
        except ValueError as error:
            result = self._state_update([], before, before, covered - offset, 'context_limit')
            return {**result, 'context_error': str(error)[:500]}
        selected = [system, memory_message(updated), *history[current:]]
        after = self.count(selected)
        reason = 'context_limit' if after > self.config.input_budget else None
        result = self._state_update([] if reason else selected, before, after, current - offset, reason)
        return {**result, 'task_memory': updated}

    @staticmethod
    def _state_update(messages, before, after, dropped, reason=None) -> dict:
        return {
            'model_messages': messages,
            'context_tokens_before': before,
            'context_tokens': after,
            'context_dropped_messages': dropped,
            'stop_reason': reason,
            'context_error': None,
        }
