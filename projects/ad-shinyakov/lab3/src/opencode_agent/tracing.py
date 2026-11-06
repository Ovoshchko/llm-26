import json
import os
from contextlib import nullcontext
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes

from .config import LAB_ROOT
from .native_tracing import NativeToolTracing


class TurnContext:
    def __init__(self, path):
        self.path = Path(path)

    def read(self):
        return json.loads(self.path.read_text()) if self.path.is_file() else {}

    def write(self, value):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value))
        temporary.replace(self.path)

    def clear(self):
        self.path.unlink(missing_ok=True)

    @classmethod
    def current(cls):
        path = os.environ.get('OPENCODE_TURN_FILE')
        return cls(path).read() if path else {}


class OpenCodeTracing:
    def __init__(self, session_id, metadata=None):
        load_dotenv(LAB_ROOT / '.env')
        self.client = get_client()
        self.session_id = session_id
        self.metadata = metadata or {}

    def start(self, question):
        return self.client.start_as_current_observation(name='opencode.turn', as_type='agent', input=question,
                                                       metadata=self.metadata, version='opencode-server-v3')

    def attributes(self):
        return propagate_attributes(session_id=self.session_id, user_id='lab-evaluation', metadata=self.metadata,
                                    tags=['ready-agent', 'opencode-server-v3', 'budget-8192', 'steps-16',
                                          'glm4-opencode:9b', self.metadata.get('run_id', 'interactive')])

    def context(self, span):
        return {'trace_id': span.trace_id, 'parent_span_id': span.id, 'session_id': self.session_id,
                'metadata': self.metadata}

    def finish(self, span, result):
        span.update(output=result['answer'], metadata={key: value for key, value in result.items()
                                                     if key not in ('events', 'answer')})
        if result['timeout'] or not result['answer']:
            span.update(level='ERROR', status_message='OpenCode завершился без ответа или превысил лимит')
        elif result['step_limit_reached']:
            span.update(level='WARNING', status_message='Достигнут предел шагов OpenCode')
        self.client.flush()

    def native_tools(self, span, events):
        NativeToolTracing().record(span.trace_id, span.id, events)


class RemoteObservation:
    def __init__(self, context, name, kind, **values):
        load_dotenv(LAB_ROOT / '.env')
        self.client = get_client()
        self.context = context or {}
        self.parent_for_tools = kind == 'generation' and values.get('metadata', {}).get('has_tools', False)
        parent = self._parent(self.context, kind)
        self.observation = self.client.start_as_current_observation(name=name, as_type=kind,
                                                                   trace_context=parent, **values)

    @staticmethod
    def _parent(context, kind):
        if not context:
            return None
        parent = context.get('generation_id') if kind == 'tool' else None
        return {'trace_id': context['trace_id'], 'parent_span_id': parent or context['parent_span_id']}

    def __enter__(self):
        self.attributes = propagate_attributes(session_id=self.context.get('session_id'), user_id='lab-evaluation',
                                               metadata=self.context.get('metadata', {}),
                                               tags=['ready-agent', 'opencode-server-v3', 'glm4-opencode:9b'])
        self.span = self.observation.__enter__()
        self.attributes.__enter__()
        if self.parent_for_tools:
            self._set_generation()
        return self.span

    def _set_generation(self):
        path = self.context.get('context_file')
        if path:
            store = TurnContext(path)
            current = store.read()
            if current.get('trace_id') == self.context.get('trace_id'):
                store.write({**current, 'generation_id': self.span.id})

    def __exit__(self, exc_type, exc_value, traceback):
        self.attributes.__exit__(exc_type, exc_value, traceback)
        self.observation.__exit__(exc_type, exc_value, traceback)
        self.client.flush()


def tool_observation(name, arguments, context=None):
    context = TurnContext.current() if context is None else context
    return RemoteObservation(context, name, 'tool', input=arguments,
                              metadata={'mcp_server': 'lab'}) if context else nullcontext()
