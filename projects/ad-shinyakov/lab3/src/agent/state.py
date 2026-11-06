from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState


class AgentState(MessagesState, total=False):
    steps: int
    context_tokens: int
    context_error: str | None
    context_tokens_before: int
    context_dropped_messages: int
    model_messages: list[AnyMessage]
    task_memory: dict
    pending_tool_calls: list[dict]
    started_at: float
    stop_reason: str | None
