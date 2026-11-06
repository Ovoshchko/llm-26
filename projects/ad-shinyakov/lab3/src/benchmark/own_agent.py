from contextlib import ExitStack
from functools import partial
from time import monotonic
from uuid import uuid4

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from agent import AgentConfig, AgentNodes
from agent.context import ContextManager
from agent.dialogue import DialogueSession
from agent.memory import MemoryCompressor, memory_message
from agent.tools import AgentTools

from .cluster import LAB_ROOT
from .recording import ToolRecording


class ObservedMemory(MemoryCompressor):
    def compress(self, history, end, *, fits=None):
        with get_client().start_as_current_observation(name='context.memory', input={
                'messages': [message.model_dump(mode='json') for message in history[:end]]}) as span:
            result = super().compress(history, end, fits=fits)
            span.update(output=result, metadata={'covered_from': 0, 'covered_until': end})
            return result


class ObservedContext(ContextManager):
    def prepare(self, messages, memory=None, config=None):
        offset = memory['covered_messages'] if memory else 0
        selected = [SystemMessage(self.system_prompt), *([memory_message(memory)] if memory else []),
                    *messages[offset:]]
        name = 'context.compress' if self.count(selected) > self.config.input_budget else 'context.budget'
        with get_client().start_as_current_observation(name=name) as span:
            result = super().prepare(messages, memory, config)
            span.update(output={key: value for key, value in result.items() if key != 'model_messages'},
                        metadata={'limit_tokens': self.config.input_budget,
                                  'before_tokens': result['context_tokens_before'],
                                  'after_tokens': result['context_tokens']})
            if result.get('stop_reason'):
                span.update(level='WARNING', status_message=result['stop_reason'])
            return result


class OwnAgentSession:
    def __init__(self, metadata=None):
        self.config = AgentConfig()
        self.client = get_client()
        self.stack = ExitStack()
        self.session_id = f'own-{uuid4().hex}'
        self.metadata = metadata or {}

    def __enter__(self):
        registry = self.stack.enter_context(AgentTools())
        tools = [self._tool(tool) for tool in registry.build()]
        nodes = AgentNodes(self.config, tools=tools)
        context = ObservedContext(self.config, tools=nodes.model.tools)
        context.compressor = ObservedMemory(self.config, context.counter)
        nodes.context = context
        self.dialogue = self.stack.enter_context(DialogueSession(
            nodes, database=LAB_ROOT / 'data/evaluation/dialogues.sqlite', thread_id=self.session_id))
        self.dialogue.graph = self.dialogue.graph.with_config(callbacks=[CallbackHandler()])
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stack.close()
        self.client.flush()

    @staticmethod
    def _tool(tool):
        def invoke(**arguments):
            return ToolRecording().execute(tool.name, arguments, partial(tool.func, **arguments))

        return StructuredTool(name=tool.name, description=tool.description, args_schema=tool.args_schema,
                              func=invoke, metadata=tool.metadata)

    def ask(self, question):
        started = monotonic()
        with self.client.start_as_current_observation(name='own-agent.turn', as_type='agent', input=question,
                                                       metadata=self.metadata, version='agent-v2') as span:
            with propagate_attributes(session_id=self.session_id, user_id='lab-evaluation',
                                      metadata=self.metadata, tags=['own-agent', 'agent-v2', 'budget-8192', 'steps-16',
                                                                   self.config.model, self.metadata.get('run_id', 'interactive')]):
                result = self._invoke(question, started)
                span.update(output=result['answer'], metadata={key: value for key, value in result.items()
                                                               if key not in ('messages', 'answer')})
                if result['stop_reason'] or result['tool_errors']:
                    span.update(level='WARNING', status_message=result['stop_reason'] or 'tool_error')
                result.update(trace_id=span.trace_id, trace_url=self.client.get_trace_url())
        self.client.flush()
        return result

    def _invoke(self, question, started):
        config = {'configurable': {'thread_id': self.session_id}}
        previous = self.dialogue.graph.get_state(config).values
        failure = None
        try:
            state = self.dialogue.ask(question)
        except Exception as error:
            failure = str(error)
            state = self.dialogue.graph.get_state(config).values
        result = self._result(state, len(previous.get('messages', [])), started)
        if failure:
            result.update(stop_reason='exception', error=failure)
        return result

    def _result(self, state, offset, started):
        messages = state['messages'][offset:]
        last = messages[-1]
        answer = last.content if isinstance(last, AIMessage) and not last.tool_calls else ''
        return {'session_id': self.session_id, 'answer': answer,
                'messages': [message.model_dump(mode='json') for message in messages],
                'steps': state['steps'], 'stop_reason': state['stop_reason'], 'context_error': state.get('context_error'),
                'tool_errors': [message.content for message in messages
                                if isinstance(message, ToolMessage) and message.status == 'error'],
                'context_tokens': state['context_tokens'], 'context_tokens_before': state['context_tokens_before'],
                'compressed': state.get('task_memory', {}).get('covered_messages', 0) > 0,
                'latency_seconds': monotonic() - started,
                'usage': {'input': sum((getattr(message, 'usage_metadata', None) or {}).get('input_tokens', 0)
                                        for message in messages),
                          'output': sum((getattr(message, 'usage_metadata', None) or {}).get('output_tokens', 0)
                                         for message in messages)}}
