from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


class HistoryValidator:
    def pending_calls(self, messages: Sequence[BaseMessage]) -> list[dict] | None:
        if messages and isinstance(messages[0], SystemMessage):
            if not isinstance(messages[0].content, str):
                return None
            messages = messages[1:]
        if not messages or not isinstance(messages[0], HumanMessage):
            return None
        pending = {}
        for message in messages:
            if not isinstance(message.content, str):
                return None
            if isinstance(message, ToolMessage):
                if message.tool_call_id not in pending:
                    return None
                del pending[message.tool_call_id]
            elif isinstance(message, (HumanMessage, AIMessage)):
                if pending:
                    return None
                if isinstance(message, AIMessage):
                    if message.invalid_tool_calls:
                        return None
                    ids = [call['id'] for call in message.tool_calls]
                    if any(not value for value in ids) or len(ids) != len(set(ids)):
                        return None
                    pending.update((call['id'], call) for call in message.tool_calls)
            else:
                return None
        return list(pending.values())
