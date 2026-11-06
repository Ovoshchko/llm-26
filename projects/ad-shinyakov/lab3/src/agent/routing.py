from typing import Literal

from .state import AgentState


def after_limits(state: AgentState) -> Literal['context', 'tools', 'stop']:
    if state.get('stop_reason'):
        return 'stop'
    return 'tools' if state.get('pending_tool_calls') else 'context'


def after_context(state: AgentState) -> Literal['model', 'stop']:
    return 'stop' if state.get('stop_reason') else 'model'
