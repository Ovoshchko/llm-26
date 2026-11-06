import json
import os
from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from transformers import AutoTokenizer


class TokenCounter(Protocol):
    def count(self, messages: Sequence[BaseMessage], tools: Sequence[dict] = ()) -> int: ...


class ChatTokenCounter:
    def __init__(self, model='zai-org/glm-4-9b-chat-hf',
                 revision='8599336fc6c125203efb2360bfaf4c80eef1d1bf', chat_template=None):
        self.model = model
        self.revision = revision
        self.tokenizer = None
        self.chat_template = chat_template

    def count(self, messages: Sequence[BaseMessage], tools: Sequence[dict] = ()) -> int:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, revision=self.revision, trust_remote_code=False,
                local_files_only=os.environ.get('HF_HUB_OFFLINE') == '1')
        if self.chat_template is not None:
            return len(self.tokenizer.apply_chat_template(
                self._agent_messages(messages), chat_template=self.chat_template,
                tools=list(tools), tokenize=True, add_generation_prompt=True))
        payload = self.serialize(messages)
        if tools:
            payload.insert(0, {'role': 'system', 'content': '', 'tools': list(tools)})
        return len(self.tokenizer.apply_chat_template(
            payload, tokenize=True, add_generation_prompt=True))

    @classmethod
    def _agent_messages(cls, messages):
        payload = []
        for message in messages:
            if isinstance(message, AIMessage) and message.tool_calls:
                content = ''.join(json.dumps({'name': call['name'], 'arguments': call['args']},
                                             ensure_ascii=False, separators=(',', ':')) for call in message.tool_calls)
                payload.append({'role': 'assistant', 'content': content})
            else:
                item = cls.serialize([message])[0]
                payload.append({'role': item['role'], 'content': message.content})
        return payload

    @staticmethod
    def serialize(messages):
        payload = []
        roles = {SystemMessage: 'system', HumanMessage: 'user', AIMessage: 'assistant', ToolMessage: 'observation'}
        for message in messages:
            if type(message) not in roles or not isinstance(message.content, str):
                raise ValueError('Only text system, human, assistant and tool messages are supported')
            if isinstance(message, AIMessage) and message.tool_calls:
                if message.content:
                    payload.append({'role': 'assistant', 'content': message.content, 'metadata': ''})
                for call in message.tool_calls:
                    payload.append({'role': 'assistant', 'metadata': call['name'],
                                    'content': json.dumps(call['args'], ensure_ascii=False)})
            else:
                payload.append({'role': roles[type(message)], 'content': message.content or ' ', 'metadata': ''})
        return payload
