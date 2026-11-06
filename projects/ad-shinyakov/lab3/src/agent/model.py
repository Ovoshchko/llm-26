from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_ollama import ChatOllama

from .config import AgentConfig
from .tool_response import ToolResponseFormat


class ChatModel(Protocol):
    tools: list[dict]

    def invoke(self, messages: Sequence[BaseMessage], config: RunnableConfig | None = None) -> AIMessage: ...


class OllamaModel:
    def __init__(self, config: AgentConfig, tools=()):
        self.config = config
        self.tools = [convert_to_openai_tool(tool) for tool in tools]
        self.response_format = ToolResponseFormat(self.tools) if self.tools and config.model == 'glm4-agent:9b' else None
        options = {'format': self.response_format.schema()} if self.response_format else {}
        client = ChatOllama(
            model=config.model,
            base_url=config.ollama_url,
            temperature=0,
            num_ctx=config.model_context_window,
            num_predict=config.max_output_tokens,
            client_kwargs={'timeout': config.timeout_seconds},
            **options,
        )
        self.client = client.bind_tools(self.tools) if self.tools else client

    def invoke(self, messages: Sequence[BaseMessage], config: RunnableConfig | None = None) -> AIMessage:
        response = self.client.invoke(messages, config=config)
        if not isinstance(response, AIMessage):
            raise TypeError('The model must return an AIMessage')
        if response.invalid_tool_calls:
            raise ValueError('The model returned invalid tool calls')
        if response.tool_calls:
            return response.model_copy(update={'content': ''})
        if self.response_format and response.content.lstrip().startswith('{'):
            return response.model_copy(update={'content': self.response_format.decode_answer(response.content)})
        return response
