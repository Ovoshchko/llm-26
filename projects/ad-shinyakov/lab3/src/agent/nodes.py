from collections.abc import Callable
from time import monotonic

from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from .config import AgentConfig
from .context import ContextManager
from .history import HistoryValidator
from .model import ChatModel, OllamaModel
from .state import AgentState


class AgentNodes:
    def __init__(self, config: AgentConfig | None = None, clock: Callable[[], float] = monotonic,
                 context_manager: ContextManager | None = None, *,
                 model: ChatModel | None = None, tools=()):
        self.config = config if config is not None else AgentConfig()
        self.clock = clock
        self.history = HistoryValidator()
        tools = list(tools)
        self.executor = ToolNode(tools, handle_tool_errors=True)
        self.model = model if model is not None else OllamaModel(self.config, tools)
        self.context = context_manager if context_manager is not None else ContextManager(
            self.config, tools=self.model.tools)
        if self.context.tools != self.model.tools:
            raise ValueError('Context and model must use the same tool schemas')

    def begin_turn(self, state: AgentState) -> dict:
        return {'steps': 0, 'context_tokens': 0, 'context_tokens_before': 0, 'context_error': None,
                'context_dropped_messages': 0, 'model_messages': [], 'pending_tool_calls': [],
                'started_at': self.clock(), 'stop_reason': None}

    def check_state(self, state: AgentState) -> dict:
        if state.get('stop_reason'):
            return {}
        if self.clock() - state['started_at'] >= self.config.timeout_seconds:
            return {'stop_reason': 'timeout'}
        if state['steps'] >= self.config.max_steps:
            return {'stop_reason': 'step_limit'}
        pending = self.history.pending_calls(state['messages'])
        if pending is None:
            return {'stop_reason': 'invalid_history', 'pending_tool_calls': []}
        return {'stop_reason': None, 'pending_tool_calls': pending}

    def prepare_context(self, state: AgentState, config: RunnableConfig) -> dict:
        result = self.context.prepare(state['messages'], state.get('task_memory'), config)
        if self.clock() - state['started_at'] >= self.config.timeout_seconds:
            result.update(stop_reason='timeout', model_messages=[])
        return result

    def call_model(self, state: AgentState, config: RunnableConfig) -> dict:
        if not state.get('model_messages'):
            raise ValueError('Prepare model_messages before calling the model')
        response = self.model.invoke(state['model_messages'], config=config)
        return {'messages': [response], 'steps': state['steps'] + 1,
                'pending_tool_calls': response.tool_calls}

    def call_tools(self, state: AgentState, config: RunnableConfig) -> dict:
        calls = state.get('pending_tool_calls', [])
        if not calls:
            return {}
        update = self.executor.invoke(calls, config=config)
        return {**update, 'pending_tool_calls': []}
