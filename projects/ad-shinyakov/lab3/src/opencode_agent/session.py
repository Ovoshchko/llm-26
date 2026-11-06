import json
from time import monotonic

import httpx

from .config import OpenCodeConfig
from .server import OpenCodeServer
from .tracing import OpenCodeTracing


class OpenCodeSession:
    def __init__(self, config=None, session_id=None, metadata=None, server=None):
        self.config = config or (server.config if server else OpenCodeConfig())
        self.session_id = session_id
        self.server = server
        self.owns_server = server is None
        self.metadata = metadata or {}
        self.tracing = OpenCodeTracing(session_id, self.metadata)

    def __enter__(self):
        if self.server is None:
            self.server = OpenCodeServer(self.config).__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.owns_server and self.server:
            self.server.__exit__(exc_type, exc_value, traceback)
            self.server = None

    def ask(self, question):
        if not question.strip():
            raise ValueError('Сообщение не должно быть пустым')
        if self.server is None:
            with self:
                return self.ask(question)
        with self.server.lock:
            return self._ask(question)

    def _ask(self, question):
        started = monotonic()
        if self.session_id is None:
            self.session_id = self.server.create_session(self.metadata.get('task_id', 'Kubernetes dialogue'))
        self.tracing.session_id = self.session_id
        previous = {message['info']['id'] for message in self.server.messages(self.session_id)}
        with self.tracing.start(question) as span, self.tracing.attributes():
            self.server.activate(self.tracing.context(span))
            status = self._send(question)
            try:
                history = self.server.messages(self.session_id)
            except httpx.HTTPError as error:
                history = []
                status['error'] = str(error)
            messages = [message for message in history if message['info']['id'] not in previous]
            events = self.events(messages)
            result = self._result(question, events, status, started)
            result.update(trace_id=span.trace_id, trace_url=self.tracing.client.get_trace_url(),
                          langfuse_session_id=self.session_id)
            self.tracing.native_tools(span, events)
            self.tracing.finish(span, result)
            self.server.turns.clear()
        self.tracing.client.flush()
        return result

    def _send(self, question):
        try:
            response = self.server.send(self.session_id, question)
            error = response.get('info', {}).get('error')
            return {'timeout': False, **({'error': str(error)} if error else {})}
        except httpx.TimeoutException as error:
            self.server.abort(self.session_id)
            return {'timeout': True, 'error': str(error)}
        except httpx.HTTPError as error:
            return {'timeout': False, 'error': str(error)}

    def _result(self, question, events, status, started):
        steps = sum(event['type'] == 'step_finish' and event.get('agent') != 'compaction' for event in events)
        errors = [event['part']['state'].get('error') for event in events
                  if event['type'] == 'tool_use' and event['part']['state']['status'] == 'error']
        return {'session_id': self.session_id, 'question': question, 'steps': steps, 'tool_errors': errors,
                'step_limit_reached': sum(event['type'] == 'step_finish' for event in events) >= self.config.max_steps,
                'answer': '' if status.get('error') else self.answer(events),
                'events': events, 'latency_seconds': monotonic() - started, **status,
                'stop_reason': 'provider_error' if status.get('error') else None}

    def export(self):
        if not self.session_id:
            return {}
        if self.server is None:
            with self:
                return self.export()
        return {'messages': self.server.messages(self.session_id)}

    @staticmethod
    def events(messages):
        types = {'step-start': 'step_start', 'step-finish': 'step_finish', 'tool': 'tool_use', 'text': 'text'}
        events = []
        for message in messages:
            info = message['info']
            if info['role'] != 'assistant':
                continue
            for part in message['parts']:
                if part['type'] in types:
                    time = part.get('time', {})
                    timestamp = time.get('end') or info['time'].get('completed') or info['time']['created']
                    events.append({'type': types[part['type']], 'timestamp': timestamp,
                                   'sessionID': info['sessionID'], 'agent': info.get('agent'), 'part': part})
        return events

    @staticmethod
    def answer(events):
        texts = [event['part']['text'] for event in events
                 if event['type'] == 'text' and event['part'].get('text')]
        if not texts:
            return ''
        try:
            result = json.loads(texts[-1])
        except json.JSONDecodeError:
            return texts[-1]
        return result.get('answer', texts[-1]) if isinstance(result, dict) else texts[-1]
